#!/usr/bin/env python3
"""
Key optimizations over the original:
  1. Cross-record embedding batching  – accumulate N records, embed all their
     texts in one Bedrock call per batch, then write SQL.
  2. DB bulk upsert via execute_values – one round-trip per RECORD_BATCH_SIZE
     rows instead of one per row.
  3. Vector literals use psycopg2 Adapter / list directly instead of building
     giant string SQL literals when a live DB connection is present.
  4. SQL file uses a StringIO buffer flushed in chunks, not one write() per row.
  5. Embedding worker pool is shared for the whole run, not recreated per record.
"""

import argparse, json, os, sys, time, io
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import psycopg2
from psycopg2.extras import execute_values

DEFAULT_REGION = "us-east-1"
RECORD_BATCH_SIZE = 200   # records accumulated before embedding + DB write
SQL_BUFFER_ROWS   = 500   # rows accumulated before flushing to the SQL file


def log(msg):
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",   default="output_test_2026-05-08T18-00-25.06559+05-30.json")
    p.add_argument("--output",  default="service_calls_inserts.sql")
    p.add_argument("--limit",   type=int, default=None)
    p.add_argument("--start",   type=int, default=0,
                   help="Input record index to start processing from (0-based)")
    p.add_argument("--embed",   action="store_true")
    p.add_argument("--model",   choices=["cohere","titan"], default="cohere")
    p.add_argument("--dimensions", type=int, default=1024, choices=[256,512,1024])
    p.add_argument("--batch-size", type=int, default=96,
                   help="Texts per Bedrock call (cohere max=96).")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--aws-access-key")
    p.add_argument("--aws-secret-key")
    p.add_argument("--aws-session-token")
    p.add_argument("--aws-region")
    p.add_argument("--db-url")
    p.add_argument("--db-user")
    p.add_argument("--db-password")
    p.add_argument("--db-iam-auth", action="store_true")
    p.add_argument("--db-sslrootcert")
    p.add_argument("--db-connect-timeout", type=int, default=10)
    p.add_argument("--skip-empty-materials", action="store_true")
    p.add_argument("--resume", action="store_true")
    # NEW: how many records to accumulate before embedding + writing
    p.add_argument("--record-batch", type=int, default=RECORD_BATCH_SIZE,
                   help="Records accumulated before embedding + DB flush.")
    return p.parse_args()


# ── streaming JSON ─────────────────────────────────────────────────────────────

def stream_top_level_objects(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        stack=[]; in_string=False; escape=False; buffer=[]; buffering=False
        while True:
            chunk = f.read(65536)
            if not chunk: break
            for char in chunk:
                if buffering: buffer.append(char)
                if escape:    escape=False; continue
                if char=='\\': escape=True; continue
                if char=='"':  in_string = not in_string; continue
                if in_string:  continue
                if char=='{':
                    if not buffering and len(stack)==1 and stack[-1]=='[':
                        buffering=True; buffer=[char]
                    stack.append('{')
                elif char=='}':
                    if stack: stack.pop()
                    if buffering and len(stack)==1 and stack[-1]=='[':
                        buffering=False; obj_str=''.join(buffer)
                        try: yield json.loads(obj_str)
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}", file=sys.stderr); raise
                        buffer=[]
                elif char=='[': stack.append('[')
                elif char==']':
                    if stack: stack.pop()


# ── DB helpers ─────────────────────────────────────────────────────────────────

def normalize_db_url(db_url):
    db_url = db_url.strip()
    if db_url.startswith('jdbc:'): db_url = db_url[5:]
    db_url = db_url.replace('${app.home}', os.getcwd())
    db_url = db_url.replace('${user.home}', os.path.expanduser('~'))
    return db_url

def generate_iam_token(db_url_normalized, db_user, args):
    import urllib.parse
    parsed   = urllib.parse.urlparse(db_url_normalized)
    host, port = parsed.hostname, parsed.port or 5432
    region   = args.aws_region or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
    kw = {"region_name": region}
    if args.aws_access_key:    kw["aws_access_key_id"]     = args.aws_access_key
    if args.aws_secret_key:    kw["aws_secret_access_key"] = args.aws_secret_key
    if args.aws_session_token: kw["aws_session_token"]      = args.aws_session_token
    token = boto3.client("rds", **kw).generate_db_auth_token(
        DBHostname=host, Port=port, DBUsername=db_user, Region=region)
    log(f"IAM token generated for {db_user}@{host}:{port}")
    return token

def get_db_connection(args):
    db_url = args.db_url or os.environ.get('DB_URL')
    if not db_url: return None
    db_url = normalize_db_url(db_url)
    log(f"Connecting to Postgres: {db_url}")
    kw = {'connect_timeout': args.db_connect_timeout}
    db_user = args.db_user or os.environ.get("DB_USER")
    if db_user: kw['user'] = db_user
    if args.db_iam_auth:
        if not db_user: raise ValueError("--db-user required with --db-iam-auth")
        kw['password'] = generate_iam_token(db_url, db_user, args)
    elif args.db_password:           kw['password'] = args.db_password
    elif os.environ.get("DB_PASSWORD"): kw['password'] = os.environ["DB_PASSWORD"]
    sslrootcert = args.db_sslrootcert
    if not sslrootcert:
        local = os.path.join(os.getcwd(), 'global-bundle.pem')
        if os.path.exists(local): sslrootcert = local; log(f"SSL cert: {local}")
    if sslrootcert: kw['sslrootcert'] = sslrootcert
    conn = psycopg2.connect(db_url, **kw)
    conn.autocommit = False
    log("DB connected.")
    return conn

def load_processed_ids(filename):
    ids, n = set(), 0
    if not os.path.exists(filename): return ids, n
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.lstrip()
            if not s.startswith('INSERT INTO service_calls'): continue
            n += 1
            try:
                vp = s.split('VALUES (', 1)[1].strip()
                if vp.startswith("'"):
                    eq = vp.find("'", 1)
                    if eq != -1: ids.add(vp[1:eq])
            except IndexError: pass
    return ids, n


# ── Bedrock ────────────────────────────────────────────────────────────────────

def get_bedrock_client(args):
    region = args.aws_region or os.environ.get("AWS_DEFAULT_REGION") or DEFAULT_REGION
    kw = {"service_name": "bedrock-runtime", "region_name": region}
    if args.aws_access_key:    kw["aws_access_key_id"]     = args.aws_access_key
    if args.aws_secret_key:    kw["aws_secret_access_key"] = args.aws_secret_key
    if args.aws_session_token: kw["aws_session_token"]      = args.aws_session_token
    log(f"Bedrock client in {region}")
    return boto3.client(**kw)

def _invoke_with_retry(client, model_id, body_str, max_retries=5, delay=1.0):
    for attempt in range(max_retries):
        try:
            r = client.invoke_model(body=body_str, modelId=model_id,
                                     accept="application/json", contentType="application/json")
            return json.loads(r['body'].read())
        except Exception as e:
            if any(k in str(e).lower() for k in ["throttling","limitexceeded","too many"]) \
                    and attempt < max_retries-1:
                print(f"Throttled, retry in {delay:.1f}s…"); time.sleep(delay); delay*=2; continue
            raise

def _truncate(text, model):
    limit = 2048 if model=='cohere' else 8192
    return text[:limit]

def embed_texts(client, texts, args):
    """Embed a flat list of texts; returns a list of embeddings in the same order."""
    if not texts: return []
    if args.dry_run:
        dim = args.dimensions if args.model=='titan' else 1024
        return [[0.0]*dim for _ in texts]

    if args.model == 'cohere':
        model_id = "cohere.embed-english-v3"
        results = []
        total = (len(texts) + args.batch_size - 1) // args.batch_size
        for i, start in enumerate(range(0, len(texts), args.batch_size), 1):
            batch = [_truncate(t, 'cohere') for t in texts[start:start+args.batch_size]]
            log(f"Cohere batch {i}/{total} ({len(batch)} texts)")
            body = json.dumps({"texts": batch, "input_type": "search_document"})
            res  = _invoke_with_retry(client, model_id, body)
            embs = res.get("embeddings", [])
            if len(embs) != len(batch):
                raise ValueError(f"Cohere: got {len(embs)} embs for {len(batch)} texts")
            results.extend(embs)
        return results

    # Titan – parallelized per-text with shared executor
    model_id = "amazon.titan-embed-text-v2:0"
    out = [None]*len(texts)
    def _one(idx, text):
        body = json.dumps({"inputText": _truncate(text,'titan'),
                            "dimensions": args.dimensions, "normalize": True})
        res = _invoke_with_retry(client, model_id, body)
        return idx, res.get("embedding")

    total = (len(texts) + args.batch_size - 1) // args.batch_size
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for bi, start in enumerate(range(0, len(texts), args.batch_size), 1):
            batch = list(enumerate(texts[start:start+args.batch_size], start))
            log(f"Titan batch {bi}/{total} ({len(batch)} texts)")
            for fut in as_completed([ex.submit(_one, i, t) for i,t in batch]):
                idx, emb = fut.result(); out[idx] = emb
    return out


# ── SQL helpers ────────────────────────────────────────────────────────────────

def _esc(v):
    if v is None: return 'NULL'
    t = str(v).replace('\x00', '').replace('\\','\\\\').replace("'","''")
    return f"E'{t}'"

def _ts(v):
    if v is None or (isinstance(v,str) and not v.strip()): return 'NULL'
    return f"'{v}'::timestamptz"

def _str(v):
    if v is None or (isinstance(v,str) and not v.strip()): return 'NULL'
    return _esc(v)

def _vec(emb):
    """Return a ::vector cast literal, or NULL."""
    if emb is None: return 'NULL'
    s = json.dumps(emb, ensure_ascii=False).replace('\\','\\\\').replace("'","''")
    return f"E'{s}'::vector"

COLS = [
    'service_call_id','custnmbr','adrscode','divisions',
    'service_description','service_description_emb',
    'problem_description','problem_description_emb',
    'appt_number','office_notes','summary_notes','summary_notes_emb',
    'onsite_start','onsite_stop','status','followup_reason',
    'material_description','material_description_emb',
]

def _row_to_sql_values(row):
    parts = []
    for col in COLS:
        v = row[col]
        if col.endswith('_emb'):       parts.append(_vec(v))
        elif col in ('onsite_start','onsite_stop'): parts.append(_ts(v))
        else:                          parts.append(_str(v))
    return '(' + ', '.join(parts) + ')'

_TS_COLS = {'onsite_start', 'onsite_stop'}

def _strip_nul(v):
    """Strip NUL bytes from a string. Postgres and psycopg2 both reject \x00."""
    if isinstance(v, str):
        return v.replace('\x00', '') or None
    return v

def _coerce(col, v):
    """
    Normalise a value for psycopg2 parameter binding.
    - embedding cols  → JSON string (with NULs stripped) or None
    - timestamp cols  → None when value is None/blank
    - everything else → None when value is None/blank, else NUL-stripped value
    """
    if col.endswith('_emb'):
        if v is None:
            return None
        # Embeddings are float lists — no NULs possible, but sanitise the
        # JSON string representation just in case
        s = json.dumps(v)
        return s.replace('\x00', '') if '\x00' in s else s
    # For all scalar cols, treat empty / whitespace-only strings as NULL
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    return _strip_nul(v)

def _row_to_pg_tuple(row):
    """Tuple for psycopg2 execute_values (vectors as JSON strings for pgvector)."""
    return tuple(_coerce(col, row[col]) for col in COLS)


# ── record → rows ──────────────────────────────────────────────────────────────

def _clean(v):
    """Strip NUL bytes and surrounding whitespace; return None if result is empty."""
    if v is None:
        return None
    s = str(v).replace('\x00', '').strip()
    return s if s else None

def _extract_texts(obj):
    """Return (texts_list, roles_list, appt_summaries, material_descs) for a record."""
    texts, roles = [], []
    appt_summaries, material_descs = [], []
    sd = _clean(obj.get('Service_Description'))
    if sd:
        texts.append(sd); roles.append('service')
    pd_ = _clean(obj.get('ProblemDesc'))
    if pd_:
        texts.append(pd_); roles.append('problem')
    for i, appt in enumerate(obj.get('app_notes',[])):
        sn = _clean(appt.get('SummaryNotes'))
        if sn:
            texts.append(sn); roles.append('summary')
            appt_summaries.append((i, sn))
    for i, mat in enumerate(obj.get('order_materials',[])):
        itd = _clean(mat.get('ITEMDESC'))
        if itd:
            texts.append(itd); roles.append('material')
            material_descs.append((i, itd))
    return texts, roles, appt_summaries, material_descs

def assign_embeddings(roles, embeddings, appt_summaries, material_descs):
    svc_emb = prob_emb = None
    sum_embs, mat_embs = {}, {}
    si = mi = 0
    for idx, role in enumerate(roles):
        if role == 'service':   svc_emb  = embeddings[idx]
        elif role == 'problem': prob_emb = embeddings[idx]
        elif role == 'summary':
            sum_embs[appt_summaries[si][0]] = embeddings[idx]; si += 1
        elif role == 'material':
            mat_embs[material_descs[mi][0]] = embeddings[idx]; mi += 1
    return svc_emb, prob_emb, sum_embs, mat_embs

def build_rows(obj, embeddings, roles, appt_summaries, material_descs):
    svc_id  = obj.get('Service_Call_ID') or obj.get('Service_Call_Id')
    base = dict(service_call_id=_clean(svc_id),
                custnmbr=_clean(obj.get('CUSTNMBR')),
                adrscode=_clean(obj.get('ADRSCODE')),
                divisions=_clean(obj.get('Divisions')),
                service_description=_clean(obj.get('Service_Description')),
                problem_description=_clean(obj.get('ProblemDesc')))

    svc_emb, prob_emb, sum_embs, mat_embs = \
        assign_embeddings(roles, embeddings, appt_summaries, material_descs)
    base['service_description_emb'] = svc_emb
    base['problem_description_emb'] = prob_emb

    rows = []
    for i, appt in enumerate(obj.get('app_notes', [])):
        r = {**base,
             'appt_number': _clean(appt.get('ApptNumber')),
             'office_notes': _clean(appt.get('OfficeNotes')),
             'summary_notes': _clean(appt.get('SummaryNotes')),
             'summary_notes_emb': sum_embs.get(i),
             'onsite_start': _clean(appt.get('OnsiteStartTime')),
             'onsite_stop': _clean(appt.get('OnsiteStopTime')),
             'status': _clean(appt.get('Status')),
             'followup_reason': _clean(appt.get('FollowupReason')),
             'material_description': None, 'material_description_emb': None}
        rows.append(r)
    for i, mat in enumerate(obj.get('order_materials', [])):
        r = {**base,
             'appt_number': None, 'office_notes': None,
             'summary_notes': None, 'summary_notes_emb': None,
             'onsite_start': None, 'onsite_stop': None,
             'status': None, 'followup_reason': None,
             'material_description': _clean(mat.get('ITEMDESC')),
             'material_description_emb': mat_embs.get(i)}
        rows.append(r)
    if not rows:
        rows.append({**base,
                     'appt_number':None,'office_notes':None,'summary_notes':None,
                     'summary_notes_emb':None,'onsite_start':None,'onsite_stop':None,
                     'status':None,'followup_reason':None,
                     'material_description':None,'material_description_emb':None})
    return rows


# ── DDL ────────────────────────────────────────────────────────────────────────

DDL = """\
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS service_calls (
    row_id           BIGSERIAL PRIMARY KEY,
    service_call_id  VARCHAR(50) NOT NULL,
    custnmbr         VARCHAR(50),
    adrscode         VARCHAR(50),
    divisions        VARCHAR(50),
    service_description TEXT,
    service_description_emb VECTOR(1024),
    problem_description TEXT,
    problem_description_emb VECTOR(1024),
    appt_number      VARCHAR(50),
    office_notes     TEXT,
    summary_notes    TEXT,
    summary_notes_emb VECTOR(1024),
    onsite_start     TIMESTAMPTZ,
    onsite_stop      TIMESTAMPTZ,
    status           VARCHAR(30),
    followup_reason  TEXT,
    material_description TEXT,
    material_description_emb VECTOR(1024),
    created_at       TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT sc_uq UNIQUE (service_call_id, appt_number, material_description)
);
CREATE INDEX IF NOT EXISTS sc_svc_id_idx   ON service_calls (service_call_id);
CREATE INDEX IF NOT EXISTS sc_svc_emb_idx  ON service_calls USING hnsw (service_description_emb  vector_cosine_ops);
CREATE INDEX IF NOT EXISTS sc_prob_emb_idx ON service_calls USING hnsw (problem_description_emb   vector_cosine_ops);
CREATE INDEX IF NOT EXISTS sc_sum_emb_idx  ON service_calls USING hnsw (summary_notes_emb          vector_cosine_ops);
CREATE INDEX IF NOT EXISTS sc_mat_emb_idx  ON service_calls USING hnsw (material_description_emb  vector_cosine_ops);

"""

# Idempotent schema migration: ensure row_id serial PK and composite unique constraint
# exist even on tables created by an older version of this script.
_ADD_PK_SQL = """
DO $$
BEGIN
    -- Add row_id serial column if missing.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'service_calls' AND column_name = 'row_id'
    ) THEN
        ALTER TABLE service_calls ADD COLUMN row_id BIGSERIAL;
    END IF;

    -- Ensure row_id is the primary key.
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
        WHERE c.conrelid = 'service_calls'::regclass
          AND c.contype = 'p'
          AND a.attname = 'row_id'
    ) THEN
        IF EXISTS (
            SELECT 1 FROM pg_constraint c
            JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = ANY(c.conkey)
            WHERE c.conrelid = 'service_calls'::regclass
              AND c.contype = 'p'
              AND a.attname = 'service_call_id'
        ) THEN
            ALTER TABLE service_calls DROP CONSTRAINT service_calls_pkey;
        END IF;
        ALTER TABLE service_calls ADD PRIMARY KEY (row_id);
    END IF;

    -- Add composite unique constraint for dedup if missing.
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'service_calls'::regclass AND conname = 'sc_uq'
    ) THEN
        ALTER TABLE service_calls
            ADD CONSTRAINT sc_uq UNIQUE (service_call_id, appt_number, material_description);
    END IF;
END$$;
"""

# Use column list for conflict handling so the script works even when the constraint name differs.
UPSERT_SQL = (
    "INSERT INTO service_calls (" + ", ".join(COLS) + ") VALUES %s "
    "ON CONFLICT (service_call_id, appt_number, material_description) DO NOTHING"
)

VEC_CAST = "::vector" 


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr); sys.exit(1)

    client = get_bedrock_client(args) if (args.embed and not args.dry_run) else None
    db_conn = get_db_connection(args)
    db_cur  = db_conn.cursor() if db_conn else None

    resume_ids, existing_rows = set(), 0
    append_mode = args.resume
    if args.resume:
        db_url_for_resume = args.db_url or os.environ.get('DB_URL')
        if db_conn and db_url_for_resume:
            # Fast path: query the DB for already-committed service_call_ids
            try:
                _rc = db_conn.cursor()
                _rc.execute("SELECT service_call_id FROM service_calls")
                resume_ids = {r[0] for r in _rc.fetchall()}
                _rc.close()
                existing_rows = len(resume_ids)
                log(f"Resume (DB): {len(resume_ids)} distinct service_call_ids already in DB — will skip them")
            except Exception as _re:
                log(f"Resume DB query failed ({_re}), falling back to SQL file scan")
                if os.path.exists(args.output):
                    resume_ids, existing_rows = load_processed_ids(args.output)
                    log(f"Resume (file): {len(resume_ids)} IDs, {existing_rows} rows")
        elif os.path.exists(args.output):
            resume_ids, existing_rows = load_processed_ids(args.output)
            append_mode = True
            log(f"Resume (file): {len(resume_ids)} IDs already done, {existing_rows} rows")

    mode = 'a' if (args.resume and os.path.exists(args.output)) else 'w'
    sql_buf   = io.StringIO()   # write SQL here; flush to file periodically
    sql_lines = 0

    with open(args.output, mode, encoding='utf-8', buffering=1 << 20) as out_f:

        if not append_mode:
            out_f.write(DDL)
            if db_cur:
                for stmt in DDL.split(';'):
                    s = stmt.strip()
                    if s: db_cur.execute(s)
                db_conn.commit()
                log("Schema/indexes ready in DB.")

        # Always ensure PK exists — safe on truncated tables where CREATE TABLE
        # IF NOT EXISTS was a no-op and the existing table has no PK yet.
        if db_cur:
            try:
                db_cur.execute(_ADD_PK_SQL)
                db_conn.commit()
                log("Primary key on service_call_id verified/added.")
            except Exception as pk_err:
                db_conn.rollback()
                log(f"Warning: could not ensure PK ({pk_err}). ON CONFLICT may fail.")

        # ── per-record state for cross-record batching ─────────────────────
        pending_objs   = []
        pending_texts  = []
        pending_offsets= []

        total_rows  = existing_rows
        input_n     = 0
        error_log   = []   # list of dicts, written to error report at end

        # Pre-build the DB template once (same for every batch)
        _col_tmpl = ["%s::vector" if c.endswith('_emb') else "%s" for c in COLS]
        _pg_template = "(" + ", ".join(_col_tmpl) + ")"

        def _classify_error(exc):
            """Return a short category string for an exception."""
            msg = str(exc).lower()
            if 'nul' in msg or '\\x00' in msg or '0x00' in msg:
                return 'NUL_BYTE'
            if 'invalid input syntax for type timestamp' in msg:
                return 'BAD_TIMESTAMP'
            if 'value too long' in msg or 'character varying' in msg:
                return 'VALUE_TOO_LONG'
            if 'unique' in msg or 'duplicate' in msg:
                return 'DUPLICATE_KEY'
            if 'null value' in msg and 'not-null' in msg:
                return 'NOT_NULL_VIOLATION'
            if 'throttl' in msg or 'limitexceeded' in msg:
                return 'BEDROCK_THROTTLE'
            if 'connection' in msg or 'timeout' in msg:
                return 'DB_CONNECTION'
            return 'OTHER'

        def _log_error(svc_id, input_idx, exc, raw_obj=None):
            """Record an error and log it to console."""
            category = _classify_error(exc)
            entry = {
                'service_call_id': svc_id,
                'input_record_index': input_idx,
                'error_category': category,
                'error_message': str(exc).replace('\n', ' ')[:500],
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            }
            if raw_obj is not None:
                # Snapshot lightweight fields for diagnosis (no embeddings)
                entry['custnmbr']    = raw_obj.get('CUSTNMBR')
                entry['appt_count']  = len(raw_obj.get('app_notes', []))
                entry['mat_count']   = len(raw_obj.get('order_materials', []))
                entry['record'] = raw_obj
                error_log.append(entry)
            log(f"[ERR] {svc_id} | {category} | {entry['error_message'][:120]}")

        def _write_error_report():
            if not error_log:
                return
            report_path = args.output.replace('.sql', '') + '_errors.json'
            with open(report_path, 'w', encoding='utf-8') as ef:
                json.dump(error_log, ef, indent=2, ensure_ascii=False)
            # Also write a compact CSV summary
            csv_path = args.output.replace('.sql', '') + '_errors.csv'
            with open(csv_path, 'w', encoding='utf-8') as cf:
                cf.write('service_call_id,input_record_index,error_category,error_message,timestamp\n')
                for e in error_log:
                    msg = e['error_message'].replace(',', ';').replace('\n', ' ')
                    cf.write(f"{e['service_call_id']},{e['input_record_index']},"
                             f"{e['error_category']},{msg},{e['timestamp']}\n")
            log(f"Error report: {len(error_log)} failures → {report_path} + {csv_path}")

            # Print a category summary table
            from collections import Counter
            counts = Counter(e['error_category'] for e in error_log)
            log("── Error summary ──────────────────────────────")
            for cat, n in counts.most_common():
                log(f"  {cat:<25} {n:>4} record(s)")
            log("───────────────────────────────────────────────")

        def flush_batch():
            """Embed pending_texts, build rows for all pending_objs, write SQL + DB.
            Errors are caught per-record; failed records are skipped and logged."""
            nonlocal total_rows, sql_lines

            if not pending_objs:
                return

            # ── embed all texts (batch-level — if this fails, fall back 1-by-1) ──
            try:
                all_embeddings = embed_texts(client, pending_texts, args) if pending_texts else []
            except Exception as emb_exc:
                log(f"[WARN] Batch embedding failed ({emb_exc}), retrying record-by-record")
                all_embeddings = None   # signal: re-embed per record below

            good_tuples  = []   # (svc_id, db_tuple) for successful rows
            good_sql     = []   # SQL strings for successful rows

            for (obj, texts, roles, appt_s, mat_d), (off, length) in \
                    zip(pending_objs, pending_offsets):

                svc_id = (obj.get('Service_Call_ID') or obj.get('Service_Call_Id') or 'UNKNOWN')

                try:
                    # If batch embed failed, try embedding just this record
                    if all_embeddings is None:
                        try:
                            rec_embs = embed_texts(client, texts, args) if texts else []
                        except Exception as re_exc:
                            _log_error(svc_id, input_n, re_exc, obj)
                            continue
                    else:
                        rec_embs = all_embeddings[off:off+length] if all_embeddings else []

                    rows = build_rows(obj, rec_embs, roles, appt_s, mat_d)

                    rec_tuples = []
                    rec_sql    = []
                    for row in rows:
                        rec_sql.append(
                            "INSERT INTO service_calls (" + ", ".join(COLS) + ") VALUES "
                            + _row_to_sql_values(row) + ";\n"
                        )
                        if db_cur:
                            rec_tuples.append(_row_to_pg_tuple(row))

                    # Validate tuples via mogrify before accepting them
                    if db_cur and rec_tuples:
                        for tup in rec_tuples:
                            db_cur.mogrify(_pg_template, tup)   # raises if bad

                    good_sql.extend(rec_sql)
                    good_tuples.extend(rec_tuples)
                    log(f"[OK] {svc_id} → {len(rows)} row(s)")

                except Exception as exc:
                    _log_error(svc_id, input_n, exc, obj)
                    if db_conn:
                        try: db_conn.rollback()
                        except Exception: pass
                    continue

            # ── write good SQL to buffer ─────────────────────────────────
            for s in good_sql:
                sql_buf.write(s)
                sql_lines += 1
                total_rows += 1

            if sql_lines >= SQL_BUFFER_ROWS:
                out_f.write(sql_buf.getvalue())
                sql_buf.truncate(0); sql_buf.seek(0); sql_lines = 0

            # ── bulk-upsert good rows to DB ──────────────────────────────
            if db_cur and good_tuples:
                try:
                    execute_values(db_cur, UPSERT_SQL, good_tuples,
                                   template=_pg_template, page_size=200)
                    db_conn.commit()
                    log(f"[DB] Committed {len(good_tuples)} rows")
                except Exception as db_exc:
                    db_conn.rollback()
                    log(f"[WARN] Bulk insert failed ({db_exc}), retrying row-by-row")
                    # Fall back: insert one row at a time so we skip only the bad one
                    saved = 0
                    for (obj, _, _, _, _), tup_list in zip(
                            pending_objs,
                            _group_tuples_by_obj(good_tuples, pending_objs, pending_offsets)):
                        svc_id = obj.get('Service_Call_ID') or obj.get('Service_Call_Id') or 'UNKNOWN'
                        try:
                            execute_values(db_cur, UPSERT_SQL, tup_list,
                                           template=_pg_template, page_size=200)
                            db_conn.commit()
                            saved += len(tup_list)
                        except Exception as row_exc:
                            db_conn.rollback()
                            _log_error(svc_id, input_n, row_exc, obj)
                    log(f"[DB] Row-by-row fallback: {saved} rows saved")

            pending_objs.clear(); pending_texts.clear(); pending_offsets.clear()

        def _group_tuples_by_obj(all_tuples, objs, offsets):
            """Re-split the flat good_tuples list back into per-object groups."""
            # Rebuild a mapping: each obj contributed len(rows) tuples.
            # We use the order they were appended to good_tuples.
            # Since we only added to good_tuples for successful records,
            # we need to walk objs/offsets and count rows per obj.
            result = []
            cursor = 0
            for (obj, texts, roles, appt_s, mat_d), (off, length) in zip(objs, offsets):
                # Approximate: rows = max(len(appt), len(mat), 1)
                n_appt = len(obj.get('app_notes', []))
                n_mat  = len(obj.get('order_materials', []))
                n_rows = max(n_appt + n_mat, 1)
                result.append(all_tuples[cursor:cursor + n_rows])
                cursor += n_rows
            return result

        # ── main record loop ───────────────────────────────────────────────
        for obj in stream_top_level_objects(args.input):
            input_n += 1
            # Skip until we reach start index (1-based input_n vs 0-based start)
            if input_n - 1 < args.start:
                continue
            if args.limit and (input_n - args.start) > args.limit:
                break

            svc_id = obj.get('Service_Call_ID') or obj.get('Service_Call_Id')
            if args.resume and svc_id and svc_id in resume_ids:
                continue
            if args.skip_empty_materials and not obj.get('order_materials'):
                continue

            try:
                texts, roles, appt_s, mat_d = _extract_texts(obj)
            except Exception as ext_exc:
                _log_error(svc_id or f'record_{input_n}', input_n, ext_exc, obj)
                continue

            off = len(pending_texts)
            pending_texts.extend(texts)
            pending_offsets.append((off, len(texts)))
            pending_objs.append((obj, texts, roles, appt_s, mat_d))

            if len(pending_objs) >= args.record_batch:
                flush_batch()
                log(f"Progress: {total_rows} rows after {input_n} records (start={args.start})")

        flush_batch()   # final partial batch

        # flush any remaining SQL buffer
        remaining = sql_buf.getvalue()
        if remaining:
            out_f.write(remaining)

    if db_conn:
        db_cur.close(); db_conn.close(); log("DB closed.")

    _write_error_report()
    log(f"Done. {total_rows} rows written to {args.output}.")


if __name__ == '__main__':
    main()
