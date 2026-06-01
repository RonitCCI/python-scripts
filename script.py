#!/usr/bin/env python3
"""Service call material recommendations via historical vector search HTTP server.

Exposes a POST /search endpoint accepting JSON payloads, performing multi-channel
pgvector lookups, and serving a ranked material recommendation response.

NEW: POST /report endpoint — generates a reliability report for a problem_description
     query using Python-side Bedrock embedding (no DB-side get_bedrock_embedding calls),
     eliminating timeout and NULL issues from calling the function inside PostgreSQL.

Usage:
  python3 server.py --server --port 8080 --db-url "postgresql://..."
"""

import argparse
import csv
import io
import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import boto3
import requests

try:
    import psycopg2
except ImportError:
    psycopg2 = None  # type: ignore


DEFAULT_REGION = 'us-east-1'

# Per-channel weights when merging four independent searches
WEIGHT_SERVICE_DESC = 0.30
WEIGHT_SUMMARY_NOTES = 0.30
WEIGHT_PROBLEM_DESC = 0.20
WEIGHT_MATERIAL = 0.20

CHANNEL_WEIGHTS = {
    'service_description': WEIGHT_SERVICE_DESC,
    'summary_notes': WEIGHT_SUMMARY_NOTES,
    'problem_description': WEIGHT_PROBLEM_DESC,
    'material_description': WEIGHT_MATERIAL,
}

# Legacy --blended mode (3-field single-query SQL)
BLENDED_SQL_WEIGHT_SERVICE = 0.40
BLENDED_SQL_WEIGHT_SUMMARY = 0.35
BLENDED_SQL_WEIGHT_PROBLEM = 0.25

DEFAULT_TOP_K_CALLS = 10

# Global reference container for CLI configurations across HTTP threads
CONFIG_ARGS: Optional[argparse.Namespace] = None


def log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    print(f'[{ts}] {msg}', flush=True)


def normalize_text(text: str) -> str:
    return ' '.join(text.strip().split())


# ---------------------------------------------------------------------------
# Embeddings (Bedrock Titan or local sentence-transformers)
# ---------------------------------------------------------------------------

def get_bedrock_client(args: argparse.Namespace):
    region = args.aws_region or os.environ.get('AWS_DEFAULT_REGION') or DEFAULT_REGION
    kw = {'service_name': 'bedrock-runtime', 'region_name': region}
    if args.aws_access_key:
        kw['aws_access_key_id'] = args.aws_access_key
    if args.aws_secret_key:
        kw['aws_secret_access_key'] = args.aws_secret_key
    if args.aws_session_token:
        kw['aws_session_token'] = args.aws_session_token
    log(f'Bedrock client initialized in {region}')
    return boto3.client(**kw)


def _invoke_with_retry(client, model_id: str, body_str: str, max_retries: int = 5, delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            r = client.invoke_model(
                body=body_str, modelId=model_id,
                accept='application/json', contentType='application/json',
            )
            return json.loads(r['body'].read())
        except Exception as e:
            if any(k in str(e).lower() for k in ['throttling', 'limitexceeded', 'too many']) and attempt < max_retries - 1:
                log(f'Bedrock throttle encountered, retry in {delay:.1f}s...')
                time.sleep(delay)
                delay *= 2
                continue
            raise


def embed_texts_bedrock(client, texts: List[str], args: argparse.Namespace) -> List[List[float]]:
    if not texts:
        return []

    model_id = 'amazon.titan-embed-text-v2:0'
    results: List[Optional[List[float]]] = [None] * len(texts)

    def worker(index: int, text: str):
        body = json.dumps({
            'inputText': text[:8192],
            'dimensions': args.dimensions,
            'normalize': True,
        })
        res = _invoke_with_retry(client, model_id, body)
        embedding = res.get('embedding')
        if embedding is None and isinstance(res.get('results'), list) and res['results']:
            embedding = res['results'][0].get('embedding')
        if embedding is None and isinstance(res.get('outputs'), list) and res['outputs']:
            embedding = res['outputs'][0].get('embedding')
        if embedding is None:
            raise ValueError(f'Bedrock response did not contain an embedding vector: {res}')
        return index, embedding

    with ThreadPoolExecutor(max_workers=min(8, len(texts))) as executor:
        futures = [executor.submit(worker, idx, text) for idx, text in enumerate(texts)]
        for future in as_completed(futures):
            idx, embedding = future.result()
            results[idx] = embedding

    return [emb for emb in results if emb is not None]


def embed_texts(texts: List[str], args: argparse.Namespace) -> List[List[float]]:
    if args.aws_access_key or args.aws_secret_key or os.environ.get('AWS_ACCESS_KEY_ID'):
        client = get_bedrock_client(args)
        return embed_texts_bedrock(client, texts, args)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            'No Bedrock credentials found and local package sentence-transformers is missing.'
        ) from exc

    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(texts, convert_to_numpy=False, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Query vector construction
# ---------------------------------------------------------------------------

def _clean_note_text(note: str) -> str:
    text = (note or '').replace('\x00', '').strip()
    return normalize_text(text) if text else ''


def build_combined_summary_notes(request: Dict[str, Any]) -> str:
    notes = request.get('app_notes') or []
    parts: List[str] = []
    for entry in notes:
        summary = _clean_note_text(entry.get('SummaryNotes') or '')
        if summary:
            parts.append(summary)
    return ' '.join(parts)


def build_material_query_text(request: Dict[str, Any], query_texts: Dict[str, str]) -> str:
    parts = [
        query_texts.get('service_description') or '',
        query_texts.get('summary_notes') or '',
        query_texts.get('problem_description') or '',
    ]
    combined = normalize_text(' '.join(p for p in parts if p))
    if combined:
        return combined

    order_materials = request.get('order_materials') or []
    material_parts = [
        normalize_text(m.get('ITEMDESC') or m.get('itemdesc') or '')
        for m in order_materials
    ]
    material_parts = [m for m in material_parts if m]
    if material_parts:
        return ' '.join(material_parts)

    return 'HVAC service call materials'


def build_query_texts(request: Dict[str, Any]) -> Dict[str, str]:
    service_desc = normalize_text(request.get('Service_Description') or '')
    summary_text = build_combined_summary_notes(request)
    problem_desc = normalize_text(request.get('ProblemDesc') or '')

    if not service_desc and not summary_text:
        service_desc = 'HVAC service call'

    texts = {
        'service_description': service_desc,
        'summary_notes': summary_text,
        'problem_description': problem_desc,
    }
    texts['material_context'] = build_material_query_text(request, texts)
    return texts


def build_query_vectors(request: Dict[str, Any], args: argparse.Namespace) -> Dict[str, List[float]]:
    texts = build_query_texts(request)
    fallback = texts['service_description'] or texts['summary_notes'] or 'HVAC service call'

    if args.simple_search:
        embed_inputs = [texts['summary_notes'] or fallback]
        embeddings = embed_texts(embed_inputs, args)
        return {'query_summary_vector': embeddings[0]}

    if args.blended:
        embed_inputs = [
            texts['service_description'] or fallback,
            texts['summary_notes'] or fallback,
            texts['problem_description'] or fallback,
        ]
        embeddings = embed_texts(embed_inputs, args)
        return {
            'query_service_desc_vector': embeddings[0],
            'query_summary_vector': embeddings[1],
            'query_problem_vector': embeddings[2],
        }

    embed_inputs = [
        texts['problem_description'] or texts['service_description'] or fallback,
        texts['service_description'] or fallback,
        texts['summary_notes'] or fallback,
        texts['material_context'] or fallback,
    ]
    embeddings = embed_texts(embed_inputs, args)
    return {
        'query_problem_vector': embeddings[0],
        'query_service_desc_vector': embeddings[1],
        'query_summary_vector': embeddings[2],
        'query_material_vector': embeddings[3],
    }


def vector_to_pg_literal(vec: List[float]) -> str:
    return '[' + ','.join(f'{x:.8g}' for x in vec) + ']'


# ---------------------------------------------------------------------------
# Database Linkage Layer
# ---------------------------------------------------------------------------

def normalize_db_url(db_url: str) -> str:
    db_url = db_url.strip()
    if db_url.startswith('jdbc:'):
        db_url = db_url[5:]
    db_url = db_url.replace('${app.home}', os.getcwd())
    db_url = db_url.replace('${user.home}', os.path.expanduser('~'))
    return db_url


def generate_iam_token(db_url_normalized: str, db_user: str, args: argparse.Namespace) -> str:
    import urllib.parse
    parsed = urllib.parse.urlparse(db_url_normalized)
    host, port = parsed.hostname, parsed.port or 5432
    region = args.aws_region or os.environ.get('AWS_DEFAULT_REGION') or DEFAULT_REGION
    kw: Dict[str, Any] = {'region_name': region}
    if args.aws_access_key:
        kw['aws_access_key_id'] = args.aws_access_key
    if args.aws_secret_key:
        kw['aws_secret_access_key'] = args.aws_secret_key
    if args.aws_session_token:
        kw['aws_session_token'] = args.aws_session_token
    return boto3.client('rds', **kw).generate_db_auth_token(
        DBHostname=host, Port=port, DBUsername=db_user, Region=region,
    )


def get_db_connection(args: argparse.Namespace):
    if psycopg2 is None:
        raise RuntimeError('psycopg2 library not available. Execute: pip install psycopg2-binary')

    db_url = args.db_url or os.environ.get('DB_URL')
    if not db_url:
        raise RuntimeError('Missing connection endpoint string. Pass --db-url or set DB_URL environment variable.')

    db_url = normalize_db_url(db_url)
    kw: Dict[str, Any] = {'connect_timeout': args.db_connect_timeout}
    db_user = args.db_user or os.environ.get('DB_USER')
    if db_user:
        kw['user'] = db_user
    if args.db_iam_auth:
        if not db_user:
            raise ValueError('--db-user parameter string expected when leveraging IAM authentication.')
        kw['password'] = generate_iam_token(db_url, db_user, args)
    elif args.db_password:
        kw['password'] = args.db_password
    elif os.environ.get('DB_PASSWORD'):
        kw['password'] = os.environ['DB_PASSWORD']

    sslrootcert = args.db_sslrootcert
    if not sslrootcert:
        local = os.path.join(os.getcwd(), 'global-bundle.pem')
        if os.path.exists(local):
            sslrootcert = local
    if sslrootcert:
        kw['sslrootcert'] = sslrootcert

    conn = psycopg2.connect(db_url, **kw)
    conn.autocommit = True
    return conn


MATERIAL_FILTERS = """
    material_description IS NOT NULL
    AND TRIM(material_description) <> ''
    AND UPPER(TRIM(material_description)) <> 'NULL'
    AND status = 'Complete'
"""

SEARCH_CHANNELS = (
    ('problem_description', 'problem_description_emb', 'query_problem_vector'),
    ('service_description', 'service_description_emb', 'query_service_desc_vector'),
    ('summary_notes', 'summary_notes_emb', 'query_summary_vector'),
    ('material_description', 'material_description_emb', 'query_material_vector'),
)


def _division_and_exclude_clauses(
    target_division: Optional[str],
    exclude_service_call_id: Optional[str],
    params: List[Any],
) -> Tuple[str, str]:
    division_clause = ''
    if target_division:
        division_clause = 'AND divisions = %s'
        params.append(target_division)

    exclude_clause = ''
    if exclude_service_call_id:
        exclude_clause = 'AND service_call_id <> %s'
        params.append(exclude_service_call_id)

    return division_clause, exclude_clause


def search_similar_by_embedding_column(
    conn,
    embedding_column: str,
    query_vector: List[float],
    *,
    search_channel: str,
    target_division: Optional[str],
    exclude_service_call_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    vec_literal = vector_to_pg_literal(query_vector)
    params: List[Any] = [vec_literal]
    division_clause, exclude_clause = _division_and_exclude_clauses(
        target_division, exclude_service_call_id, params,
    )
    params.append(top_k)

    sql = f"""
        SELECT
            service_call_id,
            custnmbr,
            divisions,
            service_description,
            summary_notes,
            material_description,
            (1 - ({embedding_column} <=> %s::vector)) AS similarity_score
        FROM service_calls
        WHERE {MATERIAL_FILTERS}
          AND {embedding_column} IS NOT NULL
          {division_clause}
          {exclude_clause}
        ORDER BY similarity_score DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    results = []
    for row in rows:
        record = dict(zip(columns, row))
        record['similarity_score'] = float(record['similarity_score'])
        record['search_channel'] = search_channel
        record['blended_score'] = record['similarity_score']
        results.append(record)
    return results


def run_four_channel_search(
    conn,
    query_vectors: Dict[str, List[float]],
    *,
    target_division: Optional[str],
    exclude_service_call_id: Optional[str],
    top_k: int,
) -> Dict[str, List[Dict[str, Any]]]:
    channel_results: Dict[str, List[Dict[str, Any]]] = {}
    for channel_name, emb_column, vector_key in SEARCH_CHANNELS:
        query_vector = query_vectors.get(vector_key)
        if not query_vector:
            channel_results[channel_name] = []
            continue
        channel_results[channel_name] = search_similar_by_embedding_column(
            conn,
            emb_column,
            query_vector,
            search_channel=channel_name,
            target_division=target_division,
            exclude_service_call_id=exclude_service_call_id,
            top_k=top_k,
        )
    return channel_results


def search_similar_calls_blended(
    conn,
    query_vectors: Dict[str, List[float]],
    *,
    target_division: Optional[str],
    exclude_service_call_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    svc_vec = vector_to_pg_literal(query_vectors['query_service_desc_vector'])
    sum_vec = vector_to_pg_literal(query_vectors['query_summary_vector'])
    prob_vec = vector_to_pg_literal(query_vectors['query_problem_vector'])

    division_clause = ''
    params: List[Any] = [svc_vec, sum_vec, prob_vec]
    if target_division:
        division_clause = 'AND divisions = %s'
        params.append(target_division)

    exclude_clause = ''
    if exclude_service_call_id:
        exclude_clause = 'AND service_call_id <> %s'
        params.append(exclude_service_call_id)

    params.append(top_k)

    sql = f"""
        SELECT
            service_call_id,
            custnmbr,
            divisions,
            service_description,
            summary_notes,
            material_description,
            (
                ({BLENDED_SQL_WEIGHT_SERVICE} * (1 - (service_description_emb <=> %s::vector)))
              + ({BLENDED_SQL_WEIGHT_SUMMARY} * (1 - (summary_notes_emb <=> %s::vector)))
              + ({BLENDED_SQL_WEIGHT_PROBLEM} * (1 - (problem_description_emb <=> %s::vector)))
            ) AS blended_score
        FROM service_calls
        WHERE {MATERIAL_FILTERS}
          AND service_description_emb IS NOT NULL
          AND summary_notes_emb IS NOT NULL
          AND problem_description_emb IS NOT NULL
          {division_clause}
          {exclude_clause}
        ORDER BY blended_score DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    results = []
    for row in rows:
        record = dict(zip(columns, row))
        record['blended_score'] = float(record['blended_score'])
        results.append(record)
    return results


def search_similar_calls_summary_only(
    conn,
    query_vector: List[float],
    *,
    target_division: Optional[str],
    exclude_service_call_id: Optional[str],
    top_k: int,
) -> List[Dict[str, Any]]:
    sum_vec = vector_to_pg_literal(query_vector)

    division_clause = ''
    params: List[Any] = [sum_vec]
    if target_division:
        division_clause = 'AND divisions = %s'
        params.append(target_division)

    exclude_clause = ''
    if exclude_service_call_id:
        exclude_clause = 'AND service_call_id <> %s'
        params.append(exclude_service_call_id)

    params.append(top_k)

    sql = f"""
        SELECT
            service_call_id,
            custnmbr,
            divisions,
            service_description,
            summary_notes,
            material_description,
            (1 - (summary_notes_emb <=> %s::vector)) AS blended_score
        FROM service_calls
        WHERE {MATERIAL_FILTERS}
          AND summary_notes_emb IS NOT NULL
          {division_clause}
          {exclude_clause}
        ORDER BY blended_score DESC
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    results = []
    for row in rows:
        record = dict(zip(columns, row))
        record['blended_score'] = float(record['blended_score'])
        results.append(record)
    return results


# ---------------------------------------------------------------------------
# Material Aggregation Engine
# ---------------------------------------------------------------------------

def parse_materials(material_description: Optional[str]) -> List[str]:
    if not material_description:
        return []
    text = normalize_text(str(material_description))
    if not text or text.upper() == 'NULL':
        return []
    for sep in (';', '|'):
        if sep in text:
            return [normalize_text(part) for part in text.split(sep) if normalize_text(part)]
    return [text]


def aggregate_material_suggestions(
    similar_calls: List[Dict[str, Any]],
    top_n: int,
    channel_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    weights = channel_weights or {}
    material_scores: Dict[str, float] = {}
    for row in similar_calls:
        channel = row.get('search_channel', '')
        weight = weights.get(channel, 1.0)
        score = row.get('similarity_score', row.get('blended_score', 0.0)) * weight
        for material in parse_materials(row.get('material_description')):
            material_scores[material] = material_scores.get(material, 0.0) + score

    ranked = sorted(material_scores.items(), key=lambda x: -x[1])[:top_n]
    suggestions = []
    for material, score in ranked:
        suggestions.append({
            'similarity_score': round(score, 4),
            'material_details': {
                'ITEMDESC': material,
            },
        })
    return suggestions


def combine_channel_results(
    channel_results: Dict[str, List[Dict[str, Any]]],
    top_n: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    all_rows: List[Dict[str, Any]] = []
    per_channel_counts: Dict[str, int] = {}
    for channel, rows in channel_results.items():
        per_channel_counts[channel] = len(rows)
        all_rows.extend(rows)

    suggestions = aggregate_material_suggestions(
        all_rows, top_n, channel_weights=CHANNEL_WEIGHTS,
    )
    return suggestions, all_rows, per_channel_counts


# ---------------------------------------------------------------------------
# NEW: /report endpoint — problem_desc similarity search via Python embedding
#
# Root cause of the original timeout/NULL issue:
#   Calling get_bedrock_embedding() inside PostgreSQL SQL means Aurora has to
#   make an outbound HTTP call to Bedrock on every query execution. This is
#   slow, subject to Aurora's statement_timeout, and unreliable under load.
#
# Fix:
#   Generate the embedding here in Python via boto3 (same Titan model),
#   then pass it as a pre-computed vector literal to the SQL query.
#   The DB only does a fast ANN vector distance scan — no outbound calls.
# ---------------------------------------------------------------------------

def search_by_problem_description(
    conn,
    problem_desc: str,
    args: argparse.Namespace,
    *,
    limit: int = 5,
    target_division: Optional[str] = None,
    exclude_service_call_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Generate embedding for problem_desc in Python, then run a pure pgvector
    cosine similarity search against problem_description_emb. No DB-side
    get_bedrock_embedding() call — avoids all timeout and NULL issues.
    """
    embeddings = embed_texts([problem_desc], args)
    if not embeddings:
        raise RuntimeError('Failed to generate embedding for the provided problem description.')

    vec_literal = vector_to_pg_literal(embeddings[0])
    params: List[Any] = [vec_literal, vec_literal]

    division_clause = ''
    if target_division:
        division_clause = 'AND divisions = %s'
        params.append(target_division)

    exclude_clause = ''
    if exclude_service_call_id:
        exclude_clause = 'AND service_call_id <> %s'
        params.append(exclude_service_call_id)

    params.append(limit)

    sql = f"""
        SELECT
            service_call_id,
            custnmbr,
            divisions,
            service_description,
            problem_description,
            summary_notes,
            material_description,
            status,
            1 - (problem_description_emb <=> %s::vector) AS similarity_score
        FROM service_calls
        WHERE problem_description_emb IS NOT NULL
          {division_clause}
          {exclude_clause}
        ORDER BY problem_description_emb <=> %s::vector
        LIMIT %s
    """

    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()

    results = []
    for row in rows:
        record = dict(zip(columns, row))
        record['similarity_score'] = round(float(record['similarity_score']), 6)
        results.append(record)
    return results


def make_report_response(
    problem_desc: str,
    args: argparse.Namespace,
    limit: int = 5,
    target_division: Optional[str] = None,
    exclude_service_call_id: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.time()

    conn = get_db_connection(args)
    try:
        results = search_by_problem_description(
            conn,
            problem_desc,
            args,
            limit=limit,
            target_division=target_division,
            exclude_service_call_id=exclude_service_call_id,
        )
    finally:
        conn.close()

    elapsed_ms = int((time.time() - start) * 1000)

    return {
        'report': {
            'query': problem_desc,
            'total_results': len(results),
            'latency_ms': elapsed_ms,
            'embedding_source': 'python_boto3_titan',  # NOT db-side get_bedrock_embedding
            'results': results,
        }
    }


# ---------------------------------------------------------------------------
# NEW: /diagnostic-report endpoint
# Runs the DB-side get_bedrock_embedding() SQL query N times and records
# every outcome: success, NULL result, timeout, or error — to prove the
# unreliability of calling the function inside PostgreSQL.
# ---------------------------------------------------------------------------

DIAGNOSTIC_SQL = """
WITH emb AS (
    SELECT get_bedrock_embedding(%s)::vector(1024) AS vec
)
SELECT
    sc.service_call_id,
    sc.problem_description,
    1 - (sc.problem_description_emb <=> emb.vec) AS similarity_score
FROM service_calls sc, emb
WHERE sc.problem_description_emb IS NOT NULL
ORDER BY sc.problem_description_emb <=> emb.vec
LIMIT 5
"""


def run_single_diagnostic(conn, problem_desc: str, run_index: int) -> Dict[str, Any]:
    """Run the DB-side get_bedrock_embedding query once and capture the outcome."""
    result: Dict[str, Any] = {
        'run': run_index,
        'status': None,
        'duration_ms': None,
        'rows_returned': None,
        'null_scores': None,
        'non_null_scores': None,
        'sample_scores': [],
        'error': None,
    }
    start = time.time()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '30000'")  # 30s per run
            cur.execute(DIAGNOSTIC_SQL, (problem_desc,))
            rows = cur.fetchall()
            elapsed_ms = int((time.time() - start) * 1000)

            scores = [row[2] for row in rows]
            null_count = sum(1 for s in scores if s is None)
            non_null_count = sum(1 for s in scores if s is not None)

            result['duration_ms'] = elapsed_ms
            result['rows_returned'] = len(rows)
            result['null_scores'] = null_count
            result['non_null_scores'] = non_null_count
            result['sample_scores'] = [round(float(s), 6) if s is not None else None for s in scores]

            if len(rows) == 0:
                result['status'] = 'empty_result'
            elif null_count == len(scores):
                result['status'] = 'all_null'
            elif null_count > 0:
                result['status'] = 'partial_null'
            else:
                result['status'] = 'success'

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        result['duration_ms'] = elapsed_ms
        err_str = str(e)
        if 'timeout' in err_str.lower() or 'statement_timeout' in err_str.lower() or 'canceling' in err_str.lower():
            result['status'] = 'timeout'
        else:
            result['status'] = 'error'
        result['error'] = err_str

    return result


def make_diagnostic_report(
    problem_desc: str,
    args: argparse.Namespace,
    runs: int = 100,
) -> Dict[str, Any]:
    """
    Run the DB-side get_bedrock_embedding() query `runs` times sequentially
    and produce a full reliability report showing NULLs, timeouts, errors,
    and successes — documenting why the function is unreliable inside SQL.
    """
    log(f'Starting diagnostic: {runs} runs for query="{problem_desc[:60]}..."')
    overall_start = time.time()

    conn = get_db_connection(args)
    run_results: List[Dict[str, Any]] = []

    try:
        for i in range(1, runs + 1):
            log(f'  Diagnostic run {i}/{runs}...')
            outcome = run_single_diagnostic(conn, problem_desc, i)
            run_results.append(outcome)
            log(f'  Run {i} → status={outcome["status"]} duration={outcome["duration_ms"]}ms')
            time.sleep(0.5)  # small gap between runs to avoid connection flooding
    finally:
        conn.close()

    total_ms = int((time.time() - overall_start) * 1000)

    # ── Summary stats ──
    statuses = [r['status'] for r in run_results]
    status_counts: Dict[str, int] = {}
    for s in statuses:
        status_counts[s] = status_counts.get(s, 0) + 1

    success_runs = [r for r in run_results if r['status'] == 'success']
    timeout_runs = [r for r in run_results if r['status'] == 'timeout']
    null_runs    = [r for r in run_results if r['status'] in ('all_null', 'partial_null')]
    error_runs   = [r for r in run_results if r['status'] == 'error']

    durations = [r['duration_ms'] for r in run_results if r['duration_ms'] is not None]
    avg_duration = round(sum(durations) / len(durations), 1) if durations else None
    max_duration = max(durations) if durations else None
    min_duration = min(durations) if durations else None

    reliability_pct = round((len(success_runs) / runs) * 100, 1)

    # ── Root cause analysis ──
    root_causes = []
    if timeout_runs:
        root_causes.append(
            f"TIMEOUT ({len(timeout_runs)}x): Aurora is calling Bedrock HTTP API inside the DB "
            f"engine, subject to statement_timeout. Network latency to Bedrock from within the "
            f"RDS VPC causes the SQL statement to exceed the timeout threshold."
        )
    if null_runs:
        root_causes.append(
            f"NULL SCORES ({len(null_runs)}x): get_bedrock_embedding() returned NULL — either "
            f"the EXCEPTION block swallowed a Bedrock error, or there is a vector type mismatch "
            f"between the stored embeddings and the query vector returned by the function."
        )
    if error_runs:
        root_causes.append(
            f"ERRORS ({len(error_runs)}x): Hard failures from Bedrock throttling, IAM permission "
            f"issues, or network connectivity from Aurora to the Bedrock endpoint."
        )

    recommendation = (
        "Generate embeddings in Python via boto3 BEFORE the SQL query, then pass the "
        "pre-computed vector as a literal (e.g. '[0.021, -0.043, ...]'::vector). "
        "The DB then only performs a fast ANN index scan with no outbound HTTP calls, "
        "eliminating all timeouts, NULLs, and throttling issues. "
        "Use the POST /report endpoint which implements this pattern."
    )

    return {
        'diagnostic_report': {
            'query': problem_desc,
            'total_runs': runs,
            'total_duration_ms': total_ms,
            'summary': {
                'reliability_percent': reliability_pct,
                'status_counts': status_counts,
                'success': len(success_runs),
                'timeouts': len(timeout_runs),
                'null_results': len(null_runs),
                'errors': len(error_runs),
                'avg_duration_ms': avg_duration,
                'min_duration_ms': min_duration,
                'max_duration_ms': max_duration,
            },
            'root_cause_analysis': root_causes,
            'recommendation': recommendation,
            'run_details': run_results,
        }
    }


# ---------------------------------------------------------------------------
# Dynamic Pipeline Response Orchestration
# ---------------------------------------------------------------------------

def make_search_response(
    request: Dict[str, Any],
    args: argparse.Namespace,
    top_n: int = 5,
    latency_ms: Optional[int] = None,
) -> Dict[str, Any]:
    query_texts = build_query_texts(request)
    query_vectors = build_query_vectors(request, args)

    target_division = None if args.no_division_filter else (
        request.get('Divisions') or request.get('divisions')
    )
    exclude_id = request.get('Service_Call_ID') or request.get('service_call_id')

    conn = get_db_connection(args)
    channel_results: Optional[Dict[str, List[Dict[str, Any]]]] = None
    similar_calls: List[Dict[str, Any]] = []
    per_channel_counts: Dict[str, int] = {}

    try:
        if args.simple_search:
            similar_calls = search_similar_calls_summary_only(
                conn,
                query_vectors['query_summary_vector'],
                target_division=target_division,
                exclude_service_call_id=exclude_id,
                top_k=args.top_k_calls,
            )
            for row in similar_calls:
                row['search_channel'] = 'summary_notes'
            search_mode = 'summary_only'
            query_source = query_texts['summary_notes'] or query_texts['service_description']
        elif args.blended:
            similar_calls = search_similar_calls_blended(
                conn,
                query_vectors,
                target_division=target_division,
                exclude_service_call_id=exclude_id,
                top_k=args.top_k_calls,
            )
            search_mode = 'blended'
            query_source = (
                f"service_desc={query_texts['service_description']!r}; "
                f"summary={query_texts['summary_notes'][:200]!r}; "
                f"problem={query_texts['problem_description']!r}"
            )
        else:
            channel_results = run_four_channel_search(
                conn,
                query_vectors,
                target_division=target_division,
                exclude_service_call_id=exclude_id,
                top_k=args.top_k_calls,
            )
            search_mode = 'four_channel'
            query_source = (
                f"problem={query_texts['problem_description'][:120]!r}; "
                f"service={query_texts['service_description']!r}; "
                f"notes={query_texts['summary_notes'][:120]!r}; "
                f"material_ctx={query_texts['material_context'][:120]!r}"
            )
    finally:
        conn.close()

    if channel_results is not None:
        suggestions, similar_calls, per_channel_counts = combine_channel_results(
            channel_results, top_n,
        )
    else:
        suggestions = aggregate_material_suggestions(similar_calls, top_n)

    metadata: Dict[str, Any] = {
        'query_vector_source': query_source,
        'search_mode': search_mode,
        'target_division': target_division,
        'similar_calls_considered': len(similar_calls),
        'total_suggestions_found': len(suggestions),
        'search_latency_ms': latency_ms if latency_ms is not None else 0,
    }
    if search_mode == 'four_channel':
        metadata['channel_weights'] = CHANNEL_WEIGHTS
        metadata['hits_per_channel'] = per_channel_counts
        metadata['top_k_per_channel'] = args.top_k_calls
    elif search_mode == 'blended':
        metadata['weights'] = {
            'service_description': 0.40,
            'summary_notes': 0.35,
            'problem_description': 0.25,
        }

    return {
        'Service_Call_ID': request.get('Service_Call_ID'),
        'search_metadata': metadata,
        'suggested_materials': suggestions,
    }


# ---------------------------------------------------------------------------
# HTTP Native Request Handler
# ---------------------------------------------------------------------------

class VectorSearchRequestHandler(BaseHTTPRequestHandler):
    """Handles runtime HTTP operations for vector matching and status monitoring."""

    def do_POST(self):
        # ── NEW: /report endpoint ──────────────────────────────────────────
        if self.path == '/report':
            self._handle_report()
            return

        if self.path == '/diagnostic-report':
            self._handle_diagnostic_report()
            return

        if self.path == '/diagnostic-report/csv':
            self._handle_diagnostic_report_csv()
            return

        if self.path != '/search':
            self.send_error(404, 'Endpoint Unsupported. Map POST requests to /search or /report')
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8')
            request_payload = json.loads(request_body)
        except (ValueError, UnicodeDecodeError) as err:
            self.send_error(400, f'Invalid JSON Payload Structure: {err}')
            return

        log(f"POST /search | Targeting Service_Call_ID={request_payload.get('Service_Call_ID', '?')}")
        start_time = time.time()

        try:
            response_payload = make_search_response(
                request=request_payload,
                args=CONFIG_ARGS,
                top_n=CONFIG_ARGS.top_n
            )
            elapsed_ms = int((time.time() - start_time) * 1000)
            response_payload['search_metadata']['search_latency_ms'] = elapsed_ms

            response_bytes = json.dumps(response_payload, ensure_ascii=False).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        except Exception as exc:
            log(f'Pipeline Failure: {exc}')
            self.send_error(500, f'Internal Vector Core Processing Error: {exc}')

    def _handle_report(self):
        """
        POST /report
        Accepts:
          {
            "problem_description": "Diagnose 01002 SLok @ ...",
            "limit": 5,                    // optional, default 5
            "division": "SAL-TM",          // optional
            "exclude_service_call_id": "250101-0125"  // optional
          }

        Why this is more reliable than the raw SQL query:
          - Embedding is generated in Python via boto3 (not inside PostgreSQL)
          - Vector is passed as a pre-computed literal — DB only does ANN scan
          - No Aurora statement_timeout risk from outbound Bedrock HTTP calls
          - No NULL from type mismatch between get_bedrock_embedding return type
        """
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8')
            payload = json.loads(request_body)
        except (ValueError, UnicodeDecodeError) as err:
            self.send_error(400, f'Invalid JSON: {err}')
            return

        problem_desc = payload.get('problem_description', '').strip()
        if not problem_desc:
            self.send_error(400, 'Missing required field: problem_description')
            return

        limit = int(payload.get('limit', 5))
        division = payload.get('division') or None
        exclude_id = payload.get('exclude_service_call_id') or None

        log(f'POST /report | query="{problem_desc[:80]}..." limit={limit}')

        try:
            response_payload = make_report_response(
                problem_desc=problem_desc,
                args=CONFIG_ARGS,
                limit=limit,
                target_division=division,
                exclude_service_call_id=exclude_id,
            )
            response_bytes = json.dumps(response_payload, ensure_ascii=False).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        except Exception as exc:
            log(f'/report failure: {exc}')
            self.send_error(500, f'Report generation failed: {exc}')


    def _handle_diagnostic_report(self):
        """
        POST /diagnostic-report
        Accepts:
          {
            "problem_description": "Diagnose 01002 ...",
            "runs": 10   // optional, default 10
          }
        Runs the DB-side get_bedrock_embedding() SQL query N times and
        records every outcome: success, NULL, timeout, error.
        """
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8')
            payload = json.loads(request_body)
        except (ValueError, UnicodeDecodeError) as err:
            self.send_error(400, f'Invalid JSON: {err}')
            return

        problem_desc = payload.get('problem_description', '').strip()
        if not problem_desc:
            self.send_error(400, 'Missing required field: problem_description')
            return

        runs = int(payload.get('runs', 100))
        if runs < 1 or runs > 500:
            self.send_error(400, 'runs must be between 1 and 50')
            return

        log(f'POST /diagnostic-report | runs={runs} query="{problem_desc[:60]}..."')

        try:
            response_payload = make_diagnostic_report(
                problem_desc=problem_desc,
                args=CONFIG_ARGS,
                runs=runs,
            )
            response_bytes = json.dumps(response_payload, ensure_ascii=False).encode('utf-8')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(response_bytes)))
            self.end_headers()
            self.wfile.write(response_bytes)

        except Exception as exc:
            log(f'/diagnostic-report failure: {exc}')
            self.send_error(500, f'Diagnostic report failed: {exc}')

    def _handle_diagnostic_report_csv(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            request_body = self.rfile.read(content_length).decode('utf-8')
            payload = json.loads(request_body)
        except (ValueError, UnicodeDecodeError) as err:
            self.send_error(400, f'Invalid JSON: {err}')
            return

        problem_desc = payload.get('problem_description', '').strip()
        if not problem_desc:
            self.send_error(400, 'Missing required field: problem_description')
            return

        runs = int(payload.get('runs', 100))
        if runs < 1 or runs > 500:
            self.send_error(400, 'runs must be between 1 and 50')
            return

        log(f'POST /diagnostic-report/csv | runs={runs} query="{problem_desc[:60]}..."')

        try:
            report = make_diagnostic_report(
                problem_desc=problem_desc,
                args=CONFIG_ARGS,
                runs=runs,
            )['diagnostic_report']

            buf = io.StringIO()
            writer = csv.writer(buf)

            writer.writerow(['DIAGNOSTIC REPORT - get_bedrock_embedding() Reliability'])
            writer.writerow(['Generated', time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())])
            writer.writerow(['Query', problem_desc])
            writer.writerow(['Total Runs', report['total_runs']])
            writer.writerow(['Total Duration (ms)', report['total_duration_ms']])
            writer.writerow([])

            writer.writerow(['--- SUMMARY ---'])
            writer.writerow(['Metric', 'Value'])
            s = report['summary']
            writer.writerow(['Reliability %',       str(s['reliability_percent']) + '%'])
            writer.writerow(['Successful Runs',      s['success']])
            writer.writerow(['Timeout Runs',         s['timeouts']])
            writer.writerow(['NULL Result Runs',     s['null_results']])
            writer.writerow(['Error Runs',           s['errors']])
            writer.writerow(['Avg Duration (ms)',    s['avg_duration_ms']])
            writer.writerow(['Min Duration (ms)',    s['min_duration_ms']])
            writer.writerow(['Max Duration (ms)',    s['max_duration_ms']])
            writer.writerow([])

            writer.writerow(['--- ROOT CAUSE ANALYSIS ---'])
            for i, cause in enumerate(report['root_cause_analysis'], 1):
                writer.writerow(['Cause ' + str(i), cause])
            writer.writerow(['Recommendation', report['recommendation']])
            writer.writerow([])

            writer.writerow(['--- PER-RUN DETAILS ---'])
            writer.writerow([
                'Run #', 'Status', 'Duration (ms)',
                'Rows Returned', 'NULL Scores', 'Non-NULL Scores',
                'Similarity Scores', 'Error Message'
            ])
            for r in report['run_details']:
                scores_str = ' | '.join(
                    str(sc) if sc is not None else 'NULL'
                    for sc in (r['sample_scores'] or [])
                )
                writer.writerow([
                    r['run'],
                    r['status'],
                    r['duration_ms'],
                    r['rows_returned']   if r['rows_returned']   is not None else '',
                    r['null_scores']     if r['null_scores']     is not None else '',
                    r['non_null_scores'] if r['non_null_scores'] is not None else '',
                    scores_str,
                    r['error'] or '',
                ])

            csv_bytes = buf.getvalue().encode('utf-8-sig')
            filename = 'diagnostic_report_' + time.strftime('%Y%m%d_%H%M%S') + '.csv'

            self.send_response(200)
            self.send_header('Content-Type', 'text/csv; charset=utf-8')
            self.send_header('Content-Disposition', 'attachment; filename="' + filename + '"')
            self.send_header('Content-Length', str(len(csv_bytes)))
            self.end_headers()
            self.wfile.write(csv_bytes)
            log('CSV report generated: ' + filename)

        except Exception as exc:
            log(f'/diagnostic-report/csv failure: {exc}')
            self.send_error(500, f'CSV report failed: {exc}')

    def do_GET(self):
        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'OK')
        else:
            self.send_error(404, 'Resource Path Specified Does Not Exist')

    def log_message(self, format, *args):
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Service Call Vector Search HTTP Daemon Engine')
    parser.add_argument('--server', action='store_true')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--top-n', type=int, default=5)
    parser.add_argument('--top-k-calls', type=int, default=DEFAULT_TOP_K_CALLS)
    parser.add_argument('--blended', action='store_true')
    parser.add_argument('--simple-search', action='store_true')
    parser.add_argument('--no-division-filter', action='store_true')
    parser.add_argument('--aws-access-key', default=None)
    parser.add_argument('--aws-secret-key', default=None)
    parser.add_argument('--aws-session-token', default=None)
    parser.add_argument('--aws-region', default=None)
    parser.add_argument('--dimensions', type=int, default=1024, choices=[256, 512, 1024])
    parser.add_argument('--db-url', default=None)
    parser.add_argument('--db-user', default=None)
    parser.add_argument('--db-password', default=None)
    parser.add_argument('--db-iam-auth', action='store_true')
    parser.add_argument('--db-sslrootcert', default=None)
    parser.add_argument('--db-connect-timeout', type=int, default=10)
    return parser.parse_args()


def main() -> None:
    global CONFIG_ARGS
    CONFIG_ARGS = parse_args()

    if not (CONFIG_ARGS.db_url or os.environ.get('DB_URL')):
        raise SystemExit(
            'Pipeline Error: Relational database endpoint instance configuration required.\n'
            'Provide --db-url or set DB_URL environment variable.'
        )

    if CONFIG_ARGS.server:
        mode_str = 'four_channel'
        if CONFIG_ARGS.simple_search:
            mode_str = 'summary_only'
        elif CONFIG_ARGS.blended:
            mode_str = 'blended'

        http_daemon = ThreadingHTTPServer((CONFIG_ARGS.host, CONFIG_ARGS.port), VectorSearchRequestHandler)
        log(f'HTTP daemon started | http://{CONFIG_ARGS.host}:{CONFIG_ARGS.port}')
        log(f'Endpoints: POST /search, POST /report, GET /health')
        log(f'Config: strategy={mode_str} | top_n={CONFIG_ARGS.top_n} | top_k={CONFIG_ARGS.top_k_calls}')

        try:
            http_daemon.serve_forever()
        except KeyboardInterrupt:
            log('Shutdown signal received. Closing...')
            http_daemon.server_close()
            log('Server stopped.')
    else:
        log('Warning: Started without --server flag.')
        log('Run with --server to start the HTTP daemon.')


if __name__ == '__main__':
    main()
