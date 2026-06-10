"""
Material Recommendation Engine v2 — REST API  (OPTIMIZED)
==========================================================
POST /api/v1/recommend-materials

DB Schema (key tables)
-----------------------
hvac_materials          – 104 k rows  | item_embedding (IVFFlat), usage_count
service_calls_test      – 167 k rows  | search_embedding (IVFFlat)
hvac_service_items_test – 812 k rows  | service_call_id → material_id mapping

Algorithm (2 indexed vector queries, no keyword scans)
-------------------------------------------------------
1. Generate one embedding from Service_Description + ProblemDesc + notes  (Bedrock)
2. Direct vector search on hvac_materials.item_embedding  → semantic material matches
3. Vector search on service_calls_test.search_embedding   → similar historical calls
4. Frequency boost – how often do our candidate materials appear in those similar calls?
5. Merge scores:  final = 0.50 * vector_sim + 0.35 * freq_score + 0.15 * pop_score
6. Exclude materials already in order_materials, normalise → similarity_score ∈ [0,1]

Why this is better than the previous approach
----------------------------------------------
✓  No ILIKE keyword scans (slow, imprecise, noisy)
✓  No per-keyword COUNT() pre-screening queries
✓  Both queries use IVFFlat indexes → sub-second latency
✓  Semantic matching catches synonyms ("refrigerant" ↔ "gas", "filter" ↔ "strainer")
✓  usage_count normalization handles global popularity correctly
"""

import json
import os
import re
import time
import logging

import boto3
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any, Optional
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────
# Configuration — hardcoded values
# ─────────────────────────────────────────────

DB_HOST               = "dynaplutus.cluster-c2d4scu2oia3.us-east-1.rds.amazonaws.com"
DB_PORT               = 5432
DB_NAME               = "postgres"
DB_USER               = "pgplutus"
DB_PASSWORD           = ""
AWS_REGION            = "us-east-1"
AWS_ACCESS_KEY_ID     = "AKIAWOXGIVL27YPZ3DF2"
AWS_SECRET_ACCESS_KEY = ""

BEDROCK_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

# Tuning knobs
TOP_K_MATERIAL_CANDIDATES = 30   # how many materials to pull from vector search
TOP_K_SIMILAR_CALLS       = 15   # how many historical calls to use for freq boost
TOP_K_RECOMMENDATIONS     = 15   # max results returned to caller
VECTOR_WEIGHT             = 0.50 # semantic similarity to the query
FREQ_WEIGHT               = 0.35 # historical usage in similar calls
POP_WEIGHT                = 0.15 # global catalog popularity
MIN_SIMILARITY_SCORE      = 0.10 # drop anything below 10 % of top item's score

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Flask application
# ─────────────────────────────────────────────

app = Flask(__name__)

# ─────────────────────────────────────────────
# Database connection
# ─────────────────────────────────────────────

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
        sslmode="require", connect_timeout=10,
    )


# ─────────────────────────────────────────────
# Bedrock embedding
# ─────────────────────────────────────────────

def generate_embedding(text: str) -> Optional[List[float]]:
    """Call Bedrock Titan Embeddings V2. Returns None on failure (graceful degradation)."""
    try:
        client = boto3.client(
            "bedrock-runtime", region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        body = json.dumps({"inputText": text[:8192]})
        resp = client.invoke_model(
            modelId=BEDROCK_EMBEDDING_MODEL, body=body,
            contentType="application/json", accept="application/json",
        )
        return json.loads(resp["body"].read()).get("embedding")
    except Exception as exc:
        log.error("Bedrock embedding failed: %s", exc)
        return None


def build_query_text(
    service_description: str,
    problem_desc: str,
    app_notes: List[Dict[str, Any]],
) -> str:
    """Combine service description, problem description, and technician notes for embedding."""
    parts = [service_description or "", problem_desc or ""]
    for note in app_notes or []:
        summary = (note.get("SummaryNotes") or "").strip()
        if summary and summary != "\x00":
            parts.append(summary)
    return " | ".join(p for p in parts if p.strip())


# ─────────────────────────────────────────────
# Recommendation engine
# ─────────────────────────────────────────────

class RecommendationEngine:

    def __init__(self, conn):
        self.conn = conn
        self._max_usage: Optional[int] = None

    def _get_max_usage(self) -> int:
        if self._max_usage is None:
            with self.conn.cursor() as cur:
                cur.execute("SELECT MAX(usage_count) FROM hvac_materials")
                self._max_usage = cur.fetchone()[0] or 1
        return self._max_usage

    # ── Step 1: direct material vector search ───────────────────────────────
    # Uses IVFFlat index on hvac_materials.item_embedding — very fast

    def vector_search_materials(
        self,
        embedding: List[float],
        excluded_descs: set,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Find the top_k catalog materials whose item_embedding is closest
        (cosine distance) to the query embedding.
        Returns list of dicts with material data + raw vector similarity score.
        """
        if not embedding:
            return []
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                """
                SELECT
                    material_id::text,
                    itemnmbr,
                    itemdesc,
                    item_category,
                    usage_count,
                    1 - (item_embedding <=> %s::vector) AS vector_sim
                FROM hvac_materials
                WHERE is_active = true
                  AND item_embedding IS NOT NULL
                ORDER BY item_embedding <=> %s::vector
                LIMIT %s
                """,
                (embedding, embedding, top_k),
            )
            results = []
            for row in cur.fetchall():
                # Exclude materials already ordered (case-insensitive)
                if (row["itemdesc"] or "").upper().strip() in excluded_descs:
                    continue
                results.append({
                    "material_id":  row["material_id"],
                    "itemnmbr":     row["itemnmbr"],
                    "itemdesc":     row["itemdesc"],
                    "category":     row["item_category"],
                    "usage_count":  row["usage_count"] or 0,
                    "vector_sim":   float(row["vector_sim"]),
                })
            return results

    # ── Step 2: find similar historical service calls ──────────────────────
    # Uses IVFFlat index on service_calls_test.search_embedding

    def find_similar_calls(
        self,
        exclude_id: str,
        embedding: List[float],
    ) -> List[str]:
        """Return service_call_ids of the TOP_K_SIMILAR_CALLS most similar past calls."""
        if not embedding:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT service_call_id
                FROM   service_calls_test
                WHERE  service_call_id != %s
                  AND  search_embedding IS NOT NULL
                ORDER  BY search_embedding <=> %s::vector
                LIMIT  %s
                """,
                (exclude_id, embedding, TOP_K_SIMILAR_CALLS),
            )
            return [row[0] for row in cur.fetchall()]

    # ── Step 3: frequency boost for our candidate materials ─────────────────

    def get_frequency_boost(
        self,
        candidate_material_ids: List[str],
        similar_call_ids: List[str],
    ) -> Dict[str, float]:
        """
        For each candidate material, count how many of the similar calls
        actually used it. Returns {material_id: freq_score ∈ [0,1]}.
        """
        if not candidate_material_ids or not similar_call_ids:
            return {}
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT material_id::text, COUNT(DISTINCT service_call_id) AS hit_count
                FROM   hvac_service_items_test
                WHERE  service_call_id = ANY(%s)
                  AND  material_id     = ANY(%s::uuid[])
                GROUP  BY material_id
                """,
                (similar_call_ids, candidate_material_ids),
            )
            total = len(similar_call_ids)
            return {row[0]: row[1] / total for row in cur.fetchall()}

    # ── Step 4: merge signals and rank ──────────────────────────────────────

    def merge_and_rank(
        self,
        material_candidates: List[Dict],
        freq_boost: Dict[str, float],
        max_usage: int,
    ) -> List[Dict[str, Any]]:
        """
        Combine vector_sim + freq_score + pop_score with configured weights.
        Returns sorted list, capped at TOP_K_RECOMMENDATIONS.
        """
        results = []
        for mat in material_candidates:
            mid          = mat["material_id"]
            vector_sim   = mat["vector_sim"]
            freq_score   = freq_boost.get(mid, 0.0)
            pop_score    = mat["usage_count"] / max_usage

            final_score  = (
                VECTOR_WEIGHT * vector_sim
                + FREQ_WEIGHT  * freq_score
                + POP_WEIGHT   * pop_score
            )

            reasons = []
            if freq_score > 0:
                hit_count = round(freq_score * TOP_K_SIMILAR_CALLS)
                reasons.append(f"Used in {hit_count}/{TOP_K_SIMILAR_CALLS} similar calls")
            reasons.append(f"Semantic similarity: {vector_sim:.3f}")

            results.append({**mat, "final_score": final_score, "reasons": reasons})

        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results[:TOP_K_RECOMMENDATIONS]

    # ── Full pipeline ────────────────────────────────────────────────────────

    def recommend(
        self,
        service_call_id: str,
        service_description: str,
        problem_desc: str,
        app_notes: List[Dict[str, Any]],
        order_materials: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        t0 = time.time()

        # Build query text & embed
        query_text = build_query_text(service_description, problem_desc, app_notes)
        log.info("Embedding call %s | text[:120]=%s", service_call_id, query_text[:120])
        embedding = generate_embedding(query_text)
        if embedding:
            log.info("Embedding ready (%d dims)", len(embedding))
        else:
            log.warning("No embedding — results will be frequency-only")

        # Exclusion set from request's order_materials
        excluded_descs = {
            (m.get("ITEMDESC") or "").upper().strip()
            for m in (order_materials or [])
            if (m.get("ITEMDESC") or "").strip()
        }

        max_usage = self._get_max_usage()

        # Step 1: direct material vector search
        material_candidates = self.vector_search_materials(
            embedding, excluded_descs, TOP_K_MATERIAL_CANDIDATES
        )
        log.info("Material vector candidates: %d", len(material_candidates))

        # Step 2: similar historical calls
        similar_ids = self.find_similar_calls(service_call_id, embedding)
        log.info("Similar calls: %d", len(similar_ids))

        # Step 3: frequency boost
        candidate_ids = [m["material_id"] for m in material_candidates]
        freq_boost = self.get_frequency_boost(candidate_ids, similar_ids)
        log.info("Candidates with freq boost: %d", len(freq_boost))

        # Step 4: merge & rank
        ranked = self.merge_and_rank(material_candidates, freq_boost, max_usage)

        latency_ms = int((time.time() - t0) * 1000)
        log.info("Done: %d results in %d ms", len(ranked), latency_ms)

        # Build response — dynamic normalisation (top item = 1.0)
        max_score = ranked[0]["final_score"] if ranked else 1.0
        suggested_materials = []
        for rec in ranked:
            sim = round(rec["final_score"] / max_score, 4)
            if sim < MIN_SIMILARITY_SCORE:
                continue
            suggested_materials.append({
                "similarity_score": sim,
                "material_details": {
                    "ITEMNMBR": rec.get("itemnmbr"),
                    "ITEMDESC": rec.get("itemdesc"),
                    "CATEGORY": rec.get("category"),
                    "REASONS":  rec.get("reasons", []),
                },
            })

        return {
            "Response": {
                "Service_Call_ID": service_call_id,
                "search_metadata": {
                    "query_vector_source":     service_description,
                    "total_suggestions_found": len(suggested_materials),
                    "search_latency_ms":       latency_ms,
                    "similar_calls_used":      len(similar_ids),
                    "material_candidates_evaluated": len(material_candidates),
                },
                "suggested_materials": suggested_materials,
            }
        }


# ─────────────────────────────────────────────
# POST /api/v1/recommend-materials
# ─────────────────────────────────────────────

@app.route("/api/v1/recommend-materials", methods=["POST"])
def recommend_materials():
    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    req_data = body.get("Request")
    if not isinstance(req_data, dict):
        return jsonify({"error": 'Body must contain a top-level "Request" object'}), 400

    service_call_id = (req_data.get("Service_Call_ID") or "").strip()
    if not service_call_id:
        return jsonify({"error": "Request.Service_Call_ID is required"}), 400

    service_description = (req_data.get("Service_Description") or "").strip()
    problem_desc        = (req_data.get("ProblemDesc") or "").strip()
    app_notes           = req_data.get("app_notes") or []
    order_materials     = req_data.get("order_materials") or []

    if not isinstance(app_notes, list):
        return jsonify({"error": "Request.app_notes must be an array"}), 400
    if not isinstance(order_materials, list):
        return jsonify({"error": "Request.order_materials must be an array"}), 400

    log.info("Request | call=%s | notes=%d | ordered=%d",
             service_call_id, len(app_notes), len(order_materials))

    conn = None
    try:
        conn = get_db_connection()
        engine = RecommendationEngine(conn)
        result = engine.recommend(
            service_call_id=service_call_id,
            service_description=service_description,
            problem_desc=problem_desc,
            app_notes=app_notes,
            order_materials=order_materials,
        )
        return jsonify(result), 200

    except psycopg2.OperationalError as exc:
        log.error("DB connection error: %s", exc)
        return jsonify({"error": "Database connection failed", "detail": str(exc)}), 500
    except psycopg2.Error as exc:
        log.error("DB query error: %s", exc)
        return jsonify({"error": "Database query failed", "detail": str(exc)}), 500
    except Exception as exc:
        log.exception("Unexpected error")
        return jsonify({"error": "Internal server error", "detail": str(exc)}), 500
    finally:
        if conn:
            conn.close()


# ─────────────────────────────────────────────
# Health checks
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/health/deep", methods=["GET"])
def health_deep():
    try:
        conn = get_db_connection()
        conn.cursor().execute("SELECT 1")
        conn.close()
        return jsonify({"status": "ok", "db": "connected"}), 200
    except Exception as exc:
        return jsonify({"status": "degraded", "db": str(exc)}), 503


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    host  = os.getenv("API_HOST", "0.0.0.0")
    port  = int(os.getenv("API_PORT", "8080"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info("Starting on %s:%d  debug=%s", host, port, debug)
    app.run(host=host, port=port, debug=debug)
