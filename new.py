"""
HVAC Service Call → pgvector Pipeline
======================================
Target DB  : AWS Aurora PostgreSQL / RDS cluster named "dyna-plutus"
Embeddings : Amazon Bedrock – amazon.titan-embed-text-v2:0  (1024-dim)
Vector ext : pgvector

Pipeline steps
--------------
1.  Connect to the DB (reads creds from env / AWS Secrets Manager).
2.  Bootstrap schema + pgvector extension.
3.  Seed materials_catalog from every unique order_material in the JSON.
4.  Generate item_embedding for every catalog row (Titan V2 via Bedrock).
5.  Upsert each service_call record.
6.  Concat service_description + all summary_notes → combined_notes.
7.  Generate search_embedding for the service call (Titan V2 via Bedrock).
8.  Upsert service_call_materials (junction) rows.
9.  Create IVFFlat indexes once the catalog is populated.

Prerequisites
-------------
    pip install psycopg2-binary boto3 python-dotenv

Environment variables (or .env file)
--------------------------------------
    DB_HOST       = <dyna-plutus cluster endpoint>
    DB_PORT       = 5432
    DB_NAME       = <database name>
    DB_USER       = <username>
    DB_PASSWORD   = <password>          # or leave blank to use IAM auth
    AWS_REGION    = us-east-1           # region for Bedrock
    JSON_FILE     = out.json            # path to the source JSON file
"""

import json
import os
import re
import sys
import time
import logging
import uuid
from datetime import datetime
from typing import Optional

import boto3
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_batch
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

load_dotenv()

DB_HOST     = "dynaplutus.cluster-c2d4scu2oia3.us-east-1.rds.amazonaws.com"
DB_PORT     = 5432
DB_NAME     = "postgres"
DB_USER     = "pgplutus"
DB_PASSWORD = ""
AWS_REGION  = "us-east-1"
AWS_ACCESS_KEY_ID = ""
AWS_SECRET_ACCESS_KEY = ""
JSON_FILE   = "top10k_results.json"

# Amazon Titan Embed Text V2 → 1024-dimensional vectors
BEDROCK_MODEL_ID = "amazon.titan-embed-text-v2:0"
VECTOR_DIM       = 1024

# IVFFlat tuning  (rule of thumb: lists ≈ sqrt(total_rows))
IVFFLAT_LISTS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Database connection
# ─────────────────────────────────────────────

def get_db_connection() -> psycopg2.extensions.connection:
    """
    Open a psycopg2 connection to the dyna-plutus cluster.
    Falls back to IAM token auth when DB_PASSWORD is empty.
    """
    password = DB_PASSWORD
    use_iam = not password

    if use_iam:
        log.info("DB_PASSWORD not set – generating IAM auth token via boto3")
        rds_client = boto3.client("rds", region_name=AWS_REGION)
        password = rds_client.generate_db_auth_token(
            DBHostname=DB_HOST,
            Port=DB_PORT,
            DBUsername=DB_USER,
            Region=AWS_REGION,
        )
    else:
        log.info("Using DB_PASSWORD auth for %s@%s:%s", DB_USER, DB_HOST, DB_PORT)

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=password,
            sslmode="require",          # Aurora always requires SSL
            connect_timeout=10,
        )
    except psycopg2.OperationalError as exc:
        message = str(exc).lower()
        if "password authentication failed" in message:
            log.error(
                "DB auth failed for %s@%s:%s. Verify DB_PASSWORD is correct or unset it to use IAM auth.",
                DB_USER,
                DB_HOST,
                DB_PORT,
            )
        elif "authentication failed" in message:
            log.error(
                "DB authentication failed for %s@%s:%s. Check DB_USER, DB_PASSWORD, and IAM auth configuration.",
                DB_USER,
                DB_HOST,
                DB_PORT,
            )
        raise

    conn.autocommit = False
    log.info("Connected to %s/%s as %s", DB_HOST, DB_NAME, DB_USER)
    return conn


# ─────────────────────────────────────────────
# Bedrock embedding helper
# ─────────────────────────────────────────────

_bedrock_client = None

def get_bedrock_client():
    """Create Bedrock client using hardcoded credentials."""
    global _bedrock_client
    if _bedrock_client is None:
        log.info("Creating Bedrock client with hardcoded credentials")
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    return _bedrock_client


def embed_text(text: str) -> list[float]:
    """
    Call Amazon Titan Embed Text V2 via Bedrock.
    Uses EC2 instance IAM role for authentication.
    """
    client = get_bedrock_client()
    body   = json.dumps({
        "inputText": text[:8000],
        "dimensions": VECTOR_DIM,
        "normalize": True,
    })

    for attempt in range(3):
        try:
            response = client.invoke_model(
                modelId     = BEDROCK_MODEL_ID,
                contentType = "application/json",
                accept      = "application/json",
                body        = body,
            )
            result = json.loads(response["body"].read())
            return result["embedding"]

        except client.exceptions.ThrottlingException:
            wait = 2 ** attempt
            log.warning("Bedrock throttled – retrying in %ss (attempt %d/3)", wait, attempt + 1)
            time.sleep(wait)
        except Exception as e:
            log.error("Bedrock embed error: %s", str(e))
            raise

    raise RuntimeError("embed_text: exceeded retry limit")


def vector_literal(embedding: list[float]) -> str:
    """
    Convert a Python list of floats to the pgvector literal string,
    e.g. '[0.021, -0.043, ...]'
    """
    return "[" + ",".join(f"{v:.8f}" for v in embedding) + "]"


# ─────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────

DDL = f"""
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── TABLE 1: hvac_materials ───────────────────────────────────────────────
-- One row per unique HVAC material / part.
-- item_embedding is vectorized at INSERT time from:
--     itemnmbr || ' ' || itemdesc || ' ' || item_category
CREATE TABLE IF NOT EXISTS hvac_materials (
    material_id     UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    itemnmbr        TEXT,
    itemdesc        TEXT        NOT NULL,
    item_category   TEXT,
    unit_price      NUMERIC(10, 2),
    usage_count     INT         NOT NULL DEFAULT 0,
    is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
    item_embedding  VECTOR({VECTOR_DIM}),           -- ← vectorized column
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Unique constraint so we don't duplicate catalog entries
CREATE UNIQUE INDEX IF NOT EXISTS uq_hvac_materials_desc
    ON hvac_materials (LOWER(itemdesc));

-- ── TABLE 2: service_calls_test ────────────────────────────────────────────────
-- One row per service call from the JSON.
-- search_embedding is vectorized from:
--     service_description || ' ' || combined_notes
CREATE TABLE IF NOT EXISTS service_calls_test (
    service_call_id     TEXT        PRIMARY KEY,
    service_description TEXT,
    problem_desc        TEXT,
    custnmbr            TEXT,
    adrscode            TEXT,
    division            TEXT,
    combined_notes      TEXT,                       -- all summary_notes joined
    search_embedding    VECTOR({VECTOR_DIM}),       -- ← vectorized column
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── TABLE 3: hvac_appointments_test ───────────────────────────────────
-- Each appointment note from app_notes; no vectors needed here.
CREATE TABLE IF NOT EXISTS hvac_appointments_test (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    service_call_id     TEXT        NOT NULL REFERENCES service_calls_test(service_call_id) ON DELETE CASCADE,
    appt_number         TEXT,
    office_notes        TEXT,
    summary_notes       TEXT,
    onsite_start_time   TIMESTAMPTZ,
    onsite_stop_time    TIMESTAMPTZ,
    status              TEXT,
    followup_reason     TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_appts_service_call_id
    ON hvac_appointments_test (service_call_id);

-- ── TABLE 4: hvac_service_items_test (junction) ────────────────────────────────
-- Links a service call to zero or more catalog materials.
-- was_recommended = TRUE  → material was AI-suggested
-- was_recommended = FALSE → material came from the original JSON order
CREATE TABLE IF NOT EXISTS hvac_service_items_test (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    service_call_id     TEXT        NOT NULL REFERENCES service_calls_test(service_call_id) ON DELETE CASCADE,
    material_id         UUID        NOT NULL REFERENCES hvac_materials(material_id),
    trxqty              NUMERIC(10, 5),
    billing_amount      NUMERIC(10, 5),
    similarity_score    FLOAT,
    was_recommended     BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scm_service_call_id
    ON hvac_service_items_test (service_call_id);
CREATE INDEX IF NOT EXISTS idx_scm_material_id
    ON hvac_service_items_test (material_id);
"""


def bootstrap_schema(conn: psycopg2.extensions.connection) -> None:
    log.info("Bootstrapping schema …")
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    log.info("Schema ready.")


# ─────────────────────────────────────────────
# IVFFlat index creation
# ─────────────────────────────────────────────

def create_vector_indexes(conn: psycopg2.extensions.connection) -> None:
    """
    IVFFlat indexes for approximate nearest-neighbour search.
    Must be created AFTER rows are inserted (IVFFlat trains on existing data).
    Use HNSW if you prefer an index that supports incremental inserts better,
    but IVFFlat is the standard choice for batch-loaded catalogs.

    Cosine distance operator class: vector_cosine_ops
    Inner-product operator class  : vector_ip_ops  (for normalised vectors)

    Because we set normalize=True in Titan V2, cosine and inner-product are
    equivalent. We use cosine_ops here for clarity.
    """
    indexes = [
        (
            "idx_hvac_materials_ivfflat",
            f"""
            CREATE INDEX IF NOT EXISTS idx_hvac_materials_ivfflat
            ON hvac_materials
            USING ivfflat (item_embedding vector_cosine_ops)
            WITH (lists = {IVFFLAT_LISTS});
            """
        ),
        (
            "idx_service_calls_test_ivfflat",
            f"""
            CREATE INDEX IF NOT EXISTS idx_service_calls_test_ivfflat
            ON service_calls_test
            USING ivfflat (search_embedding vector_cosine_ops)
            WITH (lists = {IVFFLAT_LISTS});
            """
        ),
    ]

    log.info("Creating IVFFlat vector indexes …")
    with conn.cursor() as cur:
        for name, sql in indexes:
            log.info("  → %s", name)
            cur.execute(sql)
    conn.commit()
    log.info("Vector indexes created.")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def clean_null_bytes(value: Optional[str]) -> Optional[str]:
    """Remove PostgreSQL-illegal null bytes (\\x00) from strings."""
    if value is None:
        return None
    return value.replace("\x00", "").strip() or None


def parse_timestamp(raw: Optional[str]) -> Optional[datetime]:
    """
    Parse the timestamp strings found in the JSON.
    They look like:  '2025-01-08 10:30:00.000'  or  '2025-01-08 10:30:00.30'
    Normalise milliseconds to exactly 3 digits before parsing.
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    # Replace colon-separated sub-seconds (10:30:00:000) with dot notation
    raw = re.sub(r"(\d{2}:\d{2}:\d{2}):(\d+)$", r"\1.\2", raw)
    # Pad or truncate milliseconds to 3 digits
    raw = re.sub(r"\.(\d+)$", lambda m: "." + m.group(1)[:3].ljust(3, "0"), raw)
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        try:
            return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            log.debug("Could not parse timestamp: %r", raw)
            return None


def build_material_embed_string(itemnmbr: Optional[str],
                                 itemdesc: str,
                                 item_category: Optional[str]) -> str:
    """
    Build the string that gets sent to Titan for a catalog row.
    Combine all three fields for richer semantic signal.
    """
    parts = [
        (itemnmbr    or "").strip(),
        (itemdesc    or "").strip(),
        (item_category or "").strip(),
    ]
    return " ".join(p for p in parts if p)


def build_service_embed_string(service_description: Optional[str],
                                combined_notes: Optional[str]) -> str:
    """
    Build the string that gets sent to Titan for a service call row.
    """
    parts = [
        (service_description or "").strip(),
        (combined_notes      or "").strip(),
    ]
    return " ".join(p for p in parts if p)


# ─────────────────────────────────────────────
# Materials catalog seeding
# ─────────────────────────────────────────────

def seed_hvac_materials(conn: psycopg2.extensions.connection,
                            records: list[dict]) -> dict[str, str]:
    """
    Extract every unique (itemdesc) from order_materials across all records,
    upsert into hvac_materials, embed if not already embedded.

    Returns a dict: lower(itemdesc) → material_id (UUID string)
    """
    log.info("Seeding hvac_materials …")

    # Collect unique materials keyed by lower(itemdesc)
    seen: dict[str, dict] = {}
    for record in records:
        for mat in record.get("order_materials", []):
            itemdesc = clean_null_bytes(mat.get("ITEMDESC", ""))
            if not itemdesc:
                continue
            key = itemdesc.lower()
            if key not in seen:
                seen[key] = {
                    "itemnmbr":      clean_null_bytes(mat.get("ITEMNMBR")) or None,
                    "itemdesc":      itemdesc,
                    "item_category": clean_null_bytes(mat.get("ItemCategory")) or None,
                }

    log.info("  Found %d unique materials in JSON", len(seen))

    catalog_map: dict[str, str] = {}   # lower(itemdesc) → material_id

    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        # Batch check existing items for the ones we found
        all_keys = list(seen.keys())
        # Processing in chunks of 500 to avoid long queries
        for i in range(0, len(all_keys), 500):
            chunk_keys = all_keys[i:i+500]
            cur.execute(
                "SELECT material_id, LOWER(itemdesc) as key, item_embedding IS NOT NULL AS has_embedding "
                "FROM hvac_materials WHERE LOWER(itemdesc) = ANY(%s)",
                (chunk_keys,)
            )
            for row in cur.fetchall():
                mid, key, has_embed = str(row['material_id']), row['key'], row['has_embedding']
                catalog_map[key] = mid
                if has_embed:
                    # Mark as embedded to skip later
                    if key in seen:
                        seen[key]['already_embedded'] = True

        for key, mat in seen.items():
            if key in catalog_map:
                material_id = catalog_map[key]
            else:
                # Insert new row if absolutely new
                cur.execute(
                    """
                    INSERT INTO hvac_materials (itemnmbr, itemdesc, item_category)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (LOWER(itemdesc)) DO UPDATE 
                    SET itemnmbr = EXCLUDED.itemnmbr, 
                        item_category = EXCLUDED.item_category,
                        updated_at = NOW()
                    RETURNING material_id
                    """,
                    (mat["itemnmbr"], mat["itemdesc"], mat["item_category"])
                )
                res = cur.fetchone()
                material_id = str(res[0])
                catalog_map[key] = material_id
                log.info("  UPSERT catalog: %s", mat["itemdesc"][:60])

            # Generate embedding if missing
            if not mat.get('already_embedded'):
                embed_str  = build_material_embed_string(
                    mat["itemnmbr"], mat["itemdesc"], mat["item_category"]
                )
                embedding  = embed_text(embed_str)
                vec_literal = vector_literal(embedding)

                cur.execute(
                    """
                    UPDATE hvac_materials
                    SET    item_embedding = %s::vector,
                           updated_at    = NOW()
                    WHERE  material_id   = %s
                    """,
                    (vec_literal, material_id)
                )
                log.info("  EMBEDDED catalog: %s", mat["itemdesc"][:60])

        conn.commit()

    log.info("hvac_materials seeded.  %d catalog entries.", len(catalog_map))
    return catalog_map


# ─────────────────────────────────────────────
# Service call insertion
# ─────────────────────────────────────────────

def upsert_service_call(cur, record: dict) -> str:
    """
    Upsert one service_call row (without embedding – embedding is set later).
    Returns service_call_id.
    """
    service_call_id     = record["Service_Call_ID"]
    service_description = clean_null_bytes(record.get("Service_Description"))
    problem_desc        = clean_null_bytes(record.get("ProblemDesc"))
    custnmbr            = clean_null_bytes(record.get("CUSTNMBR"))
    adrscode            = clean_null_bytes(record.get("ADRSCODE"))
    division            = clean_null_bytes(record.get("Divisions"))

    # Build combined_notes: join all non-empty SummaryNotes in order
    notes_parts = []
    for note in record.get("app_notes", []):
        sn = clean_null_bytes(note.get("SummaryNotes"))
        if sn:
            notes_parts.append(sn)
    combined_notes = " ".join(notes_parts) if notes_parts else None

    cur.execute(
        """
        INSERT INTO service_calls_test
            (service_call_id, service_description, problem_desc,
             custnmbr, adrscode, division, combined_notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (service_call_id) DO UPDATE
            SET service_description = EXCLUDED.service_description,
                problem_desc        = EXCLUDED.problem_desc,
                custnmbr            = EXCLUDED.custnmbr,
                adrscode            = EXCLUDED.adrscode,
                division            = EXCLUDED.division,
                combined_notes      = EXCLUDED.combined_notes,
                updated_at          = NOW()
        """,
        (service_call_id, service_description, problem_desc,
         custnmbr, adrscode, division, combined_notes)
    )
    return service_call_id, combined_notes, service_description


def upsert_appointments(cur, record: dict, service_call_id: str) -> None:
    """Insert all app_notes for a service call using execute_batch."""
    batch_data = []
    for note in record.get("app_notes", []):
        batch_data.append((
            service_call_id,
            clean_null_bytes(note.get("ApptNumber")),
            clean_null_bytes(note.get("OfficeNotes")),
            clean_null_bytes(note.get("SummaryNotes")),
            parse_timestamp(note.get("OnsiteStartTime")),
            parse_timestamp(note.get("OnsiteStopTime")),
            clean_null_bytes(note.get("Status")),
            clean_null_bytes(note.get("FollowupReason")),
        ))
    
    if batch_data:
        execute_batch(cur, """
            INSERT INTO hvac_appointments_test
                (service_call_id, appt_number, office_notes, summary_notes,
                 onsite_start_time, onsite_stop_time, status, followup_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, batch_data)


def upsert_hvac_service_items(cur, record: dict,
                                   service_call_id: str,
                                   catalog_map: dict[str, str]) -> None:
    """
    Insert order_materials as hvac_service_items_test rows linking to catalog using execute_batch.
    was_recommended = FALSE.
    """
    # Delete existing rows for this service call to avoid duplicates on re-run
    cur.execute(
        "DELETE FROM hvac_service_items_test WHERE service_call_id = %s AND was_recommended = FALSE",
        (service_call_id,)
    )

    batch_data = []
    material_ids_to_increment = []
    for mat in record.get("order_materials", []):
        itemdesc = clean_null_bytes(mat.get("ITEMDESC", ""))
        if not itemdesc:
            continue

        key         = itemdesc.lower()
        material_id = catalog_map.get(key)
        if not material_id:
            log.warning("  No catalog entry for '%s' – skipping junction row", itemdesc[:60])
            continue

        try:
            trxqty         = float(mat.get("TRXQTY", 0) or 0)
            billing_amount = float(mat.get("Billing_Amount", 0) or 0)
        except (ValueError, TypeError):
            trxqty, billing_amount = 0.0, 0.0

        batch_data.append((service_call_id, material_id, trxqty, billing_amount))
        material_ids_to_increment.append((material_id,))

    if batch_data:
        execute_batch(cur, """
            INSERT INTO hvac_service_items_test
                (service_call_id, material_id, trxqty, billing_amount, was_recommended)
            VALUES (%s, %s, %s, %s, FALSE)
        """, batch_data)

        # Batch increment usage_count in catalog
        execute_batch(cur, """
            UPDATE hvac_materials 
            SET usage_count = usage_count + 1, updated_at = NOW() 
            WHERE material_id = %s
        """, material_ids_to_increment)


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────

def run_pipeline(json_path: str) -> None:
    log.info("Loading JSON from %s", json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        records: list[dict] = json.load(f)
    log.info("Loaded %d service call records", len(records))

    conn = get_db_connection()

    try:
        # ── Step 1: Bootstrap schema ──────────────────────────────────────────
        bootstrap_schema(conn)

        # ── Step 2: Seed materials catalog + embed each item ─────────────────
        # (Done offline / once per unique material)
        catalog_map = seed_hvac_materials(conn, records)

        # ── Step 3: Process each service call ─────────────────────────────────
        log.info("Processing %d service call records …", len(records))

        for i, record in enumerate(records, 1):
            service_call_id = record.get("Service_Call_ID", "")
            if not service_call_id:
                log.warning("Record %d has no Service_Call_ID – skipping", i)
                continue

            with conn.cursor() as cur:
                # Check if this service call already has an embedding
                cur.execute(
                    "SELECT search_embedding IS NOT NULL FROM service_calls_test WHERE service_call_id = %s",
                    (service_call_id,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    log.info("[%d/%d] SKIP (already embedded) %s", i, len(records), service_call_id)
                    # Even if we skip embedding, we might want to ensure items are synced?
                    # For now, let's assume if it has an embedding, it's fully processed.
                    continue

                log.info("[%d/%d] %s – %s",
                         i, len(records), service_call_id,
                         record.get("Service_Description", "")[:50])

                # ── Step 3a: Upsert service_call row ─────────────────────────
                service_call_id, combined_notes, service_description = \
                    upsert_service_call(cur, record)

                # ── Step 3b: Upsert appointment notes ────────────────────────
                # Delete existing to avoid duplicates on re-run
                cur.execute(
                    "DELETE FROM hvac_appointments_test WHERE service_call_id = %s",
                    (service_call_id,)
                )
                upsert_appointments(cur, record, service_call_id)

                # ── Step 3c: Generate search_embedding for this service call ──
                embed_input = build_service_embed_string(service_description, combined_notes)

                if embed_input.strip():
                    embedding   = embed_text(embed_input)
                    vec_literal = vector_literal(embedding)
                    cur.execute(
                        """
                        UPDATE service_calls_test
                        SET    search_embedding = %s::vector,
                               updated_at       = NOW()
                        WHERE  service_call_id  = %s
                        """,
                        (vec_literal, service_call_id)
                    )
                    log.info("  Embedded service call: %s", service_call_id)
                else:
                    log.warning("  No text to embed for service call %s", service_call_id)

                # ── Step 3d: Insert junction rows (hvac_service_items) ────
                upsert_hvac_service_items(cur, record, service_call_id, catalog_map)

            conn.commit()

        # ── Step 4: Create IVFFlat indexes ────────────────────────────────────
        # Done after all rows are inserted so IVFFlat can train on the full set.
        create_vector_indexes(conn)

        log.info("Pipeline complete.")

    except Exception:
        log.exception("Pipeline failed – rolling back.")
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Optional: similarity search demo
# ─────────────────────────────────────────────

def recommend_materials(service_call_id: str, top_k: int = 5) -> None:
    """
    Demo function: given a service_call_id that is already in the DB,
    retrieve the top-K recommended materials by cosine similarity.
    """
    conn = get_db_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Fetch the stored search_embedding for this call
            cur.execute(
                "SELECT search_embedding FROM service_calls_test WHERE service_call_id = %s",
                (service_call_id,)
            )
            row = cur.fetchone()
            if not row or row["search_embedding"] is None:
                log.error("No embedding found for service call %s", service_call_id)
                return

            query_vec = row["search_embedding"]   # already a vector type in DB

            # Run ANN search against catalog
            cur.execute(
                f"""
                SELECT
                    material_id,
                    itemnmbr,
                    itemdesc,
                    item_category,
                    unit_price,
                    usage_count,
                    1 - (item_embedding <=> %s::vector) AS similarity_score
                FROM   hvac_materials
                WHERE  is_active = TRUE
                  AND  item_embedding IS NOT NULL
                ORDER  BY item_embedding <=> %s::vector
                LIMIT  %s
                """,
                (query_vec, query_vec, top_k)
            )
            results = cur.fetchall()

        log.info("\n── Top-%d material recommendations for %s ──", top_k, service_call_id)
        for rank, r in enumerate(results, 1):
            log.info(
                "  %d.  [%.4f]  %-40s  (itemnmbr: %s, category: %s, used: %dx)",
                rank,
                r["similarity_score"],
                r["itemdesc"][:40],
                r["itemnmbr"] or "—",
                r["item_category"] or "—",
                r["usage_count"],
            )
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Run the full pipeline
    run_pipeline(JSON_FILE)

    # Optional: run a demo recommendation query for the first service call
    # in the file to verify the setup end-to-end.
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as f:
            sample_id = json.load(f)[0]["Service_Call_ID"]
        recommend_materials(sample_id, top_k=5)
    except Exception as exc:
        log.warning("Demo query skipped: %s", exc)
