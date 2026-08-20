from __future__ import annotations

import json
import re
import math
from functools import lru_cache
from typing import Any

from .aligned_xsearch_service import _extract_query_encoding, _query_text
from .config import TRAIN_DATA_FILE, TRAINING_EVIDENCE_CACHE_PATH, TRAINING_EVIDENCE_INDEX_PATH
from .data_service import build_candidate_payload, build_session_payload, get_row
from .dynavis_service import build_dynavis_graph
from .user_study_aligned_service import _encode_code_row, model_token_similarity


MAX_TRAIN_EVIDENCE_SAMPLES = 0


def _display_token(token: Any) -> str:
    raw = str(token or "")
    cleaned = raw.replace("\u0120", "").replace("\u010a", "\\n")
    if cleaned.startswith("##"):
        cleaned = cleaned[2:]
    return cleaned or raw


def _terms(text: Any) -> list[str]:
    text = _display_token(text).lower()
    pieces = re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+\.\d+|\d+", text)
    terms: list[str] = []
    for piece in pieces:
        normalized = piece.strip("_").lower()
        if not normalized:
            continue
        terms.append(normalized)
        terms.extend(part for part in normalized.split("_") if part)
    return sorted(set(terms))


def _term_similarity(selected: str, candidates: list[str]) -> float:
    selected_terms = _terms(selected)
    if not selected_terms or not candidates:
        return 0.0
    candidate_set = set(candidates)
    best = 0.0
    for term in selected_terms:
        if term in candidate_set:
            best = max(best, 1.0)
            continue
        if any(term and (term in candidate or candidate in term) for candidate in candidate_set):
            best = max(best, 0.78)
    if len(selected_terms) > 1:
        overlap = sum(1 for term in selected_terms if term in candidate_set) / len(selected_terms)
        best = max(best, 0.72 * overlap)
    return round(float(min(best, 1.0)), 6)


def _safe_response(row: dict[str, Any]) -> dict[str, Any]:
    response = row.get("response")
    if isinstance(response, dict):
        return response
    if isinstance(response, str) and response.strip():
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {}
    return {}


def _concept_map(response: dict[str, Any]) -> dict[str, str]:
    concepts: dict[str, str] = {}
    for item in response.get("COMMENT_CONCEPTS", []) or []:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, str) and value.strip():
                concepts[str(key)] = value.strip()
    return concepts


def _step_map(response: dict[str, Any]) -> dict[str, dict[str, str]]:
    steps: dict[str, dict[str, str]] = {}
    for item in response.get("STEPWISE_DESCS", []) or []:
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, dict):
                steps[str(key)] = {
                    "desc": str(value.get("desc") or "").strip(),
                    "code": str(value.get("code") or "").strip(),
                }
    return steps


def _alignment_pairs(response: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in response.get("ALIGNMENT_MAP", []) or []:
        if not isinstance(item, dict):
            continue
        for concept_id, step_names in item.items():
            if not isinstance(step_names, list):
                continue
            for step_name in step_names:
                if isinstance(step_name, str):
                    pairs.append((str(concept_id), step_name))
    return pairs


def _build_training_evidence_cache() -> dict[str, Any]:
    TRAINING_EVIDENCE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    processed = 0
    with open(TRAIN_DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if MAX_TRAIN_EVIDENCE_SAMPLES and processed >= MAX_TRAIN_EVIDENCE_SAMPLES:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            train_index = int(row.get("idx", processed))
            response = _safe_response(row)
            concepts = _concept_map(response)
            steps = _step_map(response)
            pairs = _alignment_pairs(response)
            for concept_id, step_name in pairs:
                concept_text = concepts.get(concept_id, "")
                step = steps.get(step_name, {})
                step_desc = step.get("desc", "")
                step_code = step.get("code", "")
                if not concept_text or not (step_desc or step_code):
                    continue
                comment_terms = _terms(concept_text)
                code_terms = sorted(set(_terms(step_desc) + _terms(step_code)))
                if not comment_terms or not code_terms:
                    continue
                records.append(
                    {
                        "trainIndex": train_index,
                        "url": row.get("url", ""),
                        "path": row.get("path", ""),
                        "funcName": row.get("func_name", ""),
                        "conceptId": concept_id,
                        "conceptText": concept_text,
                        "stepName": step_name,
                        "stepDesc": step_desc,
                        "stepCode": step_code,
                        "commentTerms": comment_terms,
                        "codeTerms": code_terms,
                        "docstring": str(row.get("clean_docstring") or row.get("docstring") or "")[:800],
                        "code": str(row.get("original_string") or row.get("code") or "")[:1200],
                    }
                )
            processed += 1
    payload = {
        "source": "training_alignment_retrieval",
        "trainDataFile": str(TRAIN_DATA_FILE),
        "sampleLimit": MAX_TRAIN_EVIDENCE_SAMPLES or processed,
        "processedSamples": processed,
        "recordCount": len(records),
        "records": records,
    }
    with open(TRAINING_EVIDENCE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    return payload


@lru_cache(maxsize=1)
def _training_evidence_cache() -> dict[str, Any]:
    if TRAINING_EVIDENCE_CACHE_PATH.exists():
        try:
            with open(TRAINING_EVIDENCE_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "source": "training_alignment_retrieval",
        "trainDataFile": str(TRAIN_DATA_FILE),
        "sampleLimit": 0,
        "processedSamples": 0,
        "recordCount": 0,
        "records": [],
        "cacheMissing": True,
    }


@lru_cache(maxsize=1)
def _training_evidence_index() -> dict[str, Any]:
    cache = _training_evidence_cache()
    if TRAINING_EVIDENCE_INDEX_PATH.exists():
        try:
            with open(TRAINING_EVIDENCE_INDEX_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "cache": cache,
                "records": list(cache.get("records", [])),
                "commentIndex": {key: set(values) for key, values in payload.get("commentIndex", {}).items()},
                "codeIndex": {key: set(values) for key, values in payload.get("codeIndex", {}).items()},
            }
        except (json.JSONDecodeError, OSError):
            pass
    comment_index: dict[str, set[int]] = {}
    code_index: dict[str, set[int]] = {}
    records = list(cache.get("records", []))
    for idx, record in enumerate(records):
        for term in record.get("commentTerms", []):
            comment_index.setdefault(str(term), set()).add(idx)
        for term in record.get("codeTerms", []):
            code_index.setdefault(str(term), set()).add(idx)
    return {
        "cache": cache,
        "records": records,
        "commentIndex": comment_index,
        "codeIndex": code_index,
    }


def _line_number_for_code_token(candidate: dict[str, Any], code_token_index: int) -> int | None:
    for line in candidate.get("codeLines", []):
        if int(code_token_index) in {int(idx) for idx in line.get("tokenIndices", [])}:
            return int(line.get("lineNumber"))
    return None


def _pair_projection_distance(test_id: str, candidate_id: str, query_token_index: int, code_token_index: int, epoch: int) -> float | None:
    try:
        graph = build_dynavis_graph(test_id, candidate_id, epoch)
    except Exception:
        return None
    nodes = {node["id"]: node for node in graph.get("nodes", [])}
    q_node = nodes.get(f"q_tok_{query_token_index}")
    c_node = nodes.get(f"c_tok_{code_token_index}")
    if not q_node or not c_node:
        return None
    return round(math.hypot(float(q_node["x"]) - float(c_node["x"]), float(q_node["y"]) - float(c_node["y"])), 6)


def _score_training_evidence(query_token: str, code_token: str, top_k: int) -> dict[str, Any]:
    indexed = _training_evidence_index()
    cache = indexed["cache"]
    records = indexed["records"]
    if cache.get("cacheMissing"):
        return {
            "source": cache.get("source", "training_alignment_retrieval"),
            "cachePath": str(TRAINING_EVIDENCE_CACHE_PATH),
            "sampleLimit": 0,
            "processedSamples": 0,
            "recordCount": 0,
            "method": "training evidence cache is missing; run scripts/build_training_evidence_cache.py first",
            "supportingSamples": [],
            "conflictingSamples": [],
        }
    query_terms = _terms(query_token)
    code_terms = _terms(code_token)
    candidate_ids: set[int] = set()
    for term in query_terms:
        candidate_ids.update(indexed["commentIndex"].get(term, set()))
    for term in code_terms:
        candidate_ids.update(indexed["codeIndex"].get(term, set()))
    supporting: list[dict[str, Any]] = []
    conflicting: list[dict[str, Any]] = []
    for record_idx in candidate_ids:
        record = records[record_idx]
        q_sim = _term_similarity(query_token, list(record.get("commentTerms", [])))
        c_sim = _term_similarity(code_token, list(record.get("codeTerms", [])))
        support_score = q_sim * c_sim
        conflict_score = max(q_sim * (1.0 - c_sim), c_sim * (1.0 - q_sim))
        base = {
            "trainIndex": record.get("trainIndex"),
            "url": record.get("url", ""),
            "path": record.get("path", ""),
            "funcName": record.get("funcName", ""),
            "conceptId": record.get("conceptId", ""),
            "conceptText": record.get("conceptText", ""),
            "stepName": record.get("stepName", ""),
            "stepDesc": record.get("stepDesc", ""),
            "stepCode": record.get("stepCode", ""),
            "docstring": record.get("docstring", ""),
            "code": record.get("code", ""),
            "querySimilarity": q_sim,
            "codeSimilarity": c_sim,
        }
        if q_sim >= 0.55 and c_sim >= 0.55:
            supporting.append(
                {
                    **base,
                    "score": round(support_score, 6),
                    "reason": "Similar query token and similar code token appear in the same annotated concept-step alignment.",
                }
            )
        elif max(q_sim, c_sim) >= 0.62 and min(q_sim, c_sim) <= 0.35:
            if q_sim >= c_sim:
                reason = "Similar query concept is aligned to a different code expression in training data."
            else:
                reason = "Similar code expression is aligned to a different query concept in training data."
            conflicting.append(
                {
                    **base,
                    "score": round(conflict_score, 6),
                    "reason": reason,
                }
            )
    supporting.sort(key=lambda item: (float(item["score"]), float(item["querySimilarity"]) + float(item["codeSimilarity"])), reverse=True)
    conflicting.sort(key=lambda item: (float(item["score"]), max(float(item["querySimilarity"]), float(item["codeSimilarity"]))), reverse=True)
    return {
        "source": cache.get("source", "training_alignment_retrieval"),
        "cachePath": str(TRAINING_EVIDENCE_CACHE_PATH),
        "sampleLimit": cache.get("sampleLimit", MAX_TRAIN_EVIDENCE_SAMPLES or cache.get("processedSamples", 0)),
        "processedSamples": cache.get("processedSamples", 0),
        "recordCount": cache.get("recordCount", len(cache.get("records", []))),
        "method": "support_score=q_sim*c_sim; conflict_score=max(q_sim*(1-c_sim), c_sim*(1-q_sim)) over annotated concept-step alignments",
        "supportingSamples": supporting[:top_k],
        "conflictingSamples": conflicting[:top_k],
    }


def build_token_pair_attribution(payload: dict[str, Any]) -> dict[str, Any]:
    test_id = str(payload.get("testId") or "")
    candidate_id = str(payload.get("candidateId") or "")
    query_token_index = int(payload.get("queryTokenIndex"))
    code_token_index = int(payload.get("codeTokenIndex"))
    epoch = int(payload.get("epoch") or 4)
    top_k = max(1, min(20, int(payload.get("topK") or 5)))

    session = build_session_payload(test_id, 5)
    candidate = build_candidate_payload(test_id, candidate_id)
    query_tokens = list(session["query"]["tokens"])
    code_tokens = list(candidate.get("codeTokens") or [])
    if not (0 <= query_token_index < len(query_tokens)):
        raise ValueError("Query token index is out of range.")
    if not (0 <= code_token_index < len(code_tokens)):
        raise ValueError("Code token index is out of range.")

    query_token = _display_token(query_tokens[query_token_index])
    code_token = _display_token(code_tokens[code_token_index])
    code_idx = int(candidate_id.replace("code_", ""))

    model_similarity = model_token_similarity(test_id, code_idx, query_token_index, code_token_index)
    query_encoding = _extract_query_encoding(_query_text(get_row(int(test_id))))
    query_score = None
    if query_encoding.scores is not None and 0 <= query_token_index < len(query_encoding.scores):
        query_score = round(float(query_encoding.scores[query_token_index].item()), 6)

    feature, _hidden, code_scores = _encode_code_row(code_idx)
    span = feature.ori2cur_pos.get(int(code_token_index))
    code_score = None
    if span:
        start, end = int(span[0]) + 1, int(span[1]) + 1
        slots = [slot for slot in range(start, end) if 0 <= slot < code_scores.shape[0]]
        if slots:
            code_score = round(float(code_scores[slots].mean().item()), 6)

    query_concept_ids = [
        int(concept["conceptId"])
        for concept in session["query"].get("concepts", [])
        if query_token_index in {int(idx) for idx in concept.get("tokenIndices", [])}
    ]
    matching_concepts = []
    line_similarity = None
    line_number = _line_number_for_code_token(candidate, code_token_index)
    for match in candidate.get("conceptMatches", []):
        q_match = query_token_index in {int(idx) for idx in match.get("queryTokenIndices", [])}
        c_match = code_token_index in {int(idx) for idx in match.get("codeTokenIndices", [])}
        if q_match or c_match:
            matching_concepts.append(
                {
                    "conceptId": int(match.get("conceptId", -1)),
                    "queryText": match.get("queryText", ""),
                    "codeText": match.get("codeText", ""),
                    "lineNumber": match.get("lineNumber"),
                    "similarity": match.get("similarity"),
                    "containsBothSelectedTokens": bool(q_match and c_match),
                }
            )
        if c_match and line_similarity is None:
            line_similarity = match.get("similarity")

    candidate_summary = next((item for item in session.get("candidates", []) if item.get("id") == candidate_id), None)
    projection_distance = _pair_projection_distance(test_id, candidate_id, query_token_index, code_token_index, epoch)
    training_evidence = _score_training_evidence(query_token, code_token, top_k)

    return {
        "status": "ok",
        "source": "xsearch_step7000_online_evidence",
        "selectedPair": {
            "queryTokenIndex": query_token_index,
            "codeTokenIndex": code_token_index,
            "queryToken": query_token,
            "codeToken": code_token,
            "modelCosine": model_similarity["cosine"],
            "modelPositiveCosine": model_similarity["positive"],
            "projectionDistance": projection_distance,
            "queryHighlightScore": query_score,
            "codeHighlightScore": code_score,
            "queryHighlighted": bool(query_score is not None and query_score >= 0.4),
            "codeHighlighted": bool(code_score is not None and code_score >= 0.5),
            "queryConceptIds": query_concept_ids,
            "lineNumber": line_number,
        },
        "candidateContext": {
            "candidateId": candidate_id,
            "codeIdx": code_idx,
            "rank": candidate_summary.get("rank") if candidate_summary else None,
            "similarity": candidate_summary.get("similarity") if candidate_summary else candidate.get("similarity"),
            "lineSimilarity": line_similarity,
            "metadata": candidate.get("metadata", {}),
            "matchingConcepts": matching_concepts[:6],
        },
        "trainingEvidence": training_evidence,
        "diagnosis": {
            "summary": "This combines current Step_7000 token-pair evidence with training-set concept-step evidence retrieval.",
            "evidenceCaveat": "Training evidence is not gradient influence; it is retrieved from annotated training alignments and ranked by lexical/token similarity.",
        },
    }
