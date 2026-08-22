"""CodeBERT compatibility adapter for ConceptLens.

This module deliberately does not share XSearch's alignment or intervention state.
It exposes a diagnostic Concept -> Block -> Line -> Token payload built from the
fine-tuned CodeBERT encoder's contextual hidden states.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import CODEBERT_BASE_PATH, CODEBERT_CHECKPOINT_PATH, CODEBERT_INDEX_PATH, CONCEPT_COLORS, RUNS_DIR, TTV_TOOL_PATH
from .data_service import build_code_lines, load_csn_codebase, load_csn_queries
from .dynavis_service import _project_vectors, _semantic_ast_blocks

MODEL_ID = "codebert"
MODEL_PAYLOAD = {
    "id": MODEL_ID,
    "name": "CodeBERT",
    "type": "generic_dual_encoder",
    "description": "Representation-level Semantic Alignment",
}
CAPABILITIES = {
    "concepts": True,
    "hierarchy": True,
    "token_similarity": True,
    "projection": True,
    "inspect": True,
    "intervention": True,
    "reranking_after_intervention": True,
    "external_effects": True,
}

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "function", "in", "into",
    "is", "it", "its", "of", "on", "or", "the", "this", "that", "to", "with", "when", "where",
    "while", "will", "does", "do", "returns", "return", "using", "use", "value", "values",
}
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]")
_MODEL_LOCK = threading.Lock()
_INDEX_LOCK = threading.Lock()
_PROJECTION_LOCK = threading.Lock()
_LOCAL_INTERVENTION_LOCK = threading.RLock()
_LOCAL_CODE_OVERRIDES: dict[tuple[str, str], dict[int, np.ndarray]] = {}
_LOCAL_INTERVENTION_REVISIONS: dict[tuple[str, str], int] = {}


def model_metadata() -> dict[str, Any]:
    return {"model": MODEL_PAYLOAD, "capabilities": CAPABILITIES}


def is_supported_test(test_id: str) -> bool:
    return str(test_id).startswith("csn_") and str(test_id)[4:].isdigit()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


@lru_cache(maxsize=1)
def _model_bundle():
    if not CODEBERT_BASE_PATH.exists():
        raise FileNotFoundError(f"CodeBERT base model is unavailable: {CODEBERT_BASE_PATH}")
    if not CODEBERT_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"CodeBERT checkpoint is unavailable: {CODEBERT_CHECKPOINT_PATH}")
    import torch
    from transformers import RobertaModel, RobertaTokenizerFast

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizerFast.from_pretrained(str(CODEBERT_BASE_PATH), local_files_only=True, add_prefix_space=True)
    encoder = RobertaModel.from_pretrained(str(CODEBERT_BASE_PATH), local_files_only=True)
    state = torch.load(CODEBERT_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    encoder_state = {key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if unexpected or missing:
        raise RuntimeError(f"CodeBERT encoder checkpoint mismatch: missing={len(missing)}, unexpected={len(unexpected)}")
    encoder.to(device).eval()
    return tokenizer, encoder, device


def _source_ranges(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text_parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        value = str(token)
        if text_parts:
            cursor += 1
        text_parts.append(value)
        ranges.append((cursor, cursor + len(value)))
        cursor += len(value)
    return " ".join(text_parts), ranges


def _encode_token_sequences(token_batches: list[list[str]], batch_size: int = 96) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return a pooled vector plus one contextual vector for every source token."""
    import torch

    tokenizer, encoder, device = _model_bundle()
    results: list[tuple[np.ndarray, np.ndarray]] = []
    with _MODEL_LOCK, torch.inference_mode():
        for start in range(0, len(token_batches), batch_size):
            batch = token_batches[start:start + batch_size]
            texts_and_ranges = [_source_ranges(tokens) for tokens in batch]
            encoded = tokenizer(
                [item[0] for item in texts_and_ranges],
                padding=True,
                truncation=True,
                max_length=256,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = encoded.pop("offset_mapping").cpu().numpy()
            encoded = {key: value.to(device) for key, value in encoded.items()}
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    hidden = encoder(**encoded).last_hidden_state.float().cpu().numpy()
            else:
                hidden = encoder(**encoded).last_hidden_state.cpu().numpy()
            masks = encoded["attention_mask"].cpu().numpy().astype(bool)
            for batch_index, (_text, ranges) in enumerate(texts_and_ranges):
                valid = masks[batch_index] & (offsets[batch_index, :, 1] > offsets[batch_index, :, 0])
                valid_hidden = hidden[batch_index][valid]
                pooled = _normalize(valid_hidden.mean(axis=0)) if len(valid_hidden) else np.zeros(hidden.shape[-1], dtype=np.float32)
                vectors: list[np.ndarray] = []
                for left, right in ranges:
                    overlaps = valid & (offsets[batch_index, :, 0] < right) & (offsets[batch_index, :, 1] > left)
                    vectors.append(_normalize(hidden[batch_index][overlaps].mean(axis=0)) if overlaps.any() else pooled.copy())
                results.append((pooled.astype(np.float32), np.stack(vectors).astype(np.float32) if vectors else np.empty((0, hidden.shape[-1]), dtype=np.float32)))
    return results


def _code_tokens(row: dict[str, Any]) -> list[str]:
    tokens = list(row.get("code_tokens") or [])
    if tokens:
        return [str(token) for token in tokens]
    return _TOKEN_RE.findall(str(row.get("code") or row.get("original_string") or ""))


def _query_row(test_id: str) -> dict[str, Any]:
    if not is_supported_test(test_id):
        raise ValueError("CodeBERT compatibility mode currently supports CSN Python examples (csn_<query index>).")
    query_index = int(str(test_id).split("_", 1)[1])
    rows = load_csn_queries()
    if query_index < 0 or query_index >= len(rows):
        raise KeyError(f"CSN query index out of range: {query_index}")
    return rows[query_index]


def _query_tokens(row: dict[str, Any]) -> list[str]:
    tokens = [str(token) for token in row.get("docstring_tokens") or []]
    return tokens or _TOKEN_RE.findall(str(row.get("clean_docstring") or row.get("docstring") or ""))


def _semantic_concepts(tokens: list[str]) -> list[dict[str, Any]]:
    spans: list[list[int]] = []
    current: list[int] = []
    for index, token in enumerate(tokens):
        normalized = token.lower().strip()
        semantic = bool(re.search(r"[a-z0-9_]", normalized)) and normalized not in _STOP_WORDS
        possessive = normalized in {"'s", "s"} and current
        if semantic or possessive:
            current.append(index)
            continue
        if current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    spans = [span for span in spans if any(len(tokens[index]) > 1 for index in span)]
    if len(spans) > 3:
        ranked = sorted(spans, key=lambda span: (sum(len(tokens[index]) for index in span), -span[0]), reverse=True)[:3]
        spans = sorted(ranked, key=lambda span: span[0])
    if not spans:
        fallback = [index for index, token in enumerate(tokens) if re.search(r"[A-Za-z0-9]", token)][:3]
        spans = [[index] for index in fallback]
    return [
        {
            "id": f"concept_{concept_id}",
            "conceptId": concept_id,
            "tokenIndices": span,
            "text": " ".join(tokens[index] for index in span),
            "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
        }
        for concept_id, span in enumerate(spans)
    ]


def _index_metadata() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CODEBERT_INDEX_PATH.exists():
        _build_code_index()
    archive = np.load(CODEBERT_INDEX_PATH, mmap_mode="r")
    return archive["vectors"], archive["code_indices"], archive["url_indices"]


def _build_code_index() -> None:
    """Build a full CSN code retrieval index once, separate from XSearch caches."""
    with _INDEX_LOCK:
        if CODEBERT_INDEX_PATH.exists():
            return
        rows = load_csn_codebase()
        vectors: list[np.ndarray] = []
        for start in range(0, len(rows), 768):
            encoded = _encode_token_sequences([_code_tokens(row) for row in rows[start:start + 768]])
            vectors.extend(item[0] for item in encoded)
        CODEBERT_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = CODEBERT_INDEX_PATH.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp_path,
            vectors=np.stack(vectors).astype(np.float32),
            code_indices=np.arange(len(rows), dtype=np.int32),
            url_indices=np.array([str(row.get("url") or "") for row in rows]),
        )
        tmp_path.replace(CODEBERT_INDEX_PATH)


def _rank_candidates(query_vector: np.ndarray, query_row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    vectors, code_indices, urls = _index_metadata()
    scores = np.asarray(vectors @ query_vector, dtype=np.float32)
    count = min(max(top_k, 20), len(scores))
    selected = np.argpartition(scores, -count)[-count:]
    selected = selected[np.argsort(scores[selected])[::-1]]
    target_url = str(query_row.get("url") or "")
    gt_positions = np.flatnonzero(urls == target_url)
    gt_index = int(code_indices[int(gt_positions[0])]) if len(gt_positions) else None
    rank_order = np.argsort(scores)[::-1]
    result = []
    for rank, position in enumerate(selected, start=1):
        code_index = int(code_indices[position])
        result.append({"id": f"code_{code_index}", "codeIdx": code_index, "rank": rank, "similarity": round(float(scores[position]), 6), "isGroundTruth": code_index == gt_index})
    if gt_index is not None and all(item["codeIdx"] != gt_index for item in result):
        gt_position = int(np.flatnonzero(rank_order == gt_index)[0])
        result.append({"id": f"code_{gt_index}", "codeIdx": gt_index, "rank": gt_position + 1, "similarity": round(float(scores[gt_index]), 6), "isGroundTruth": True})
    return result


@lru_cache(maxsize=64)
def _session_cache(test_id: str, top_k: int) -> dict[str, Any]:
    row = _query_row(test_id)
    tokens = _query_tokens(row)
    query_vector, _token_vectors = _encode_token_sequences([tokens])[0]
    concepts = _semantic_concepts(tokens)
    candidates = _rank_candidates(query_vector, row, top_k)
    return {
        "testId": str(test_id),
        **model_metadata(),
        "rankingSource": "codebert_epoch2_full_csn_dual_encoder_index",
        "conceptSource": "semantic_span_mean_pooling_over_codebert_contextual_tokens",
        "query": {
            "rawText": row.get("clean_docstring") or row.get("docstring") or "",
            "tokens": tokens,
            "concepts": concepts,
            "metadata": {key: str(row.get(key) or "") for key in ("repo", "path", "func_name", "url")},
        },
        "candidates": [
            {
                **item,
                "metadata": {
                    "repo": str(load_csn_codebase()[item["codeIdx"]].get("repo") or ""),
                    "path": str(load_csn_codebase()[item["codeIdx"]].get("path") or ""),
                    "funcName": str(load_csn_codebase()[item["codeIdx"]].get("func_name") or ""),
                    "url": str(load_csn_codebase()[item["codeIdx"]].get("url") or ""),
                },
            }
            for item in candidates
        ],
    }


def build_session_payload(test_id: str, top_k: int = 20) -> dict[str, Any]:
    return _session_cache(str(test_id), int(top_k))


def _line_by_token(code_lines: list[dict[str, Any]]) -> dict[int, int]:
    return {int(token_index): int(line["lineNumber"]) for line in code_lines for token_index in line.get("tokenIndices", [])}


def _dynavis_projection(test_id: str, candidate_id: str, vectors: np.ndarray, nodes: list[dict[str, Any]]) -> np.ndarray:
    """Create a CodeBERT-only DynaVis cache from its contextual token states."""
    revision = _LOCAL_INTERVENTION_REVISIONS.get((str(test_id), candidate_id), 0)
    run_key = hashlib.sha1(f"codebert_epoch2_dynavis_v2:{test_id}:{candidate_id}:{revision}".encode("utf-8")).hexdigest()[:16]
    content_path = RUNS_DIR / f"codebert_{run_key}"
    projection_path = content_path / "visualize" / "DynaVis_codebert" / "epochs" / "epoch_4" / "projection.npy"
    with _PROJECTION_LOCK:
        if not projection_path.exists():
            dataset_dir = content_path / "dataset"
            epochs_dir = content_path / "epochs"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            epochs_dir.mkdir(parents=True, exist_ok=True)
            with open(dataset_dir / "info.json", "w", encoding="utf-8") as handle:
                json.dump({
                    "model": "codebert-epoch2-generic-adapter",
                    "representation": "contextual_query_and_code_token_hidden_states",
                    "testId": str(test_id),
                    "candidateId": candidate_id,
                    "availableEpochs": [1, 2, 3, 4],
                    "hiddenDim": int(vectors.shape[1]),
                }, handle, ensure_ascii=False, indent=2)
            with open(dataset_dir / "nodes.json", "w", encoding="utf-8") as handle:
                json.dump(nodes, handle, ensure_ascii=False, indent=2)
            # The encoder is fixed in this diagnostic view; repeat its true states so
            # DynaVis receives a stable temporal sequence without borrowing XSearch states.
            for epoch in range(1, 5):
                epoch_dir = epochs_dir / f"epoch_{epoch}"
                epoch_dir.mkdir(parents=True, exist_ok=True)
                np.save(epoch_dir / "embeddings.npy", vectors.astype(np.float32))
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            if str(TTV_TOOL_PATH) not in sys.path:
                sys.path.insert(0, str(TTV_TOOL_PATH))
            from visualize.dynavis.runner import DynaVisRunner

            runner = DynaVisRunner(
                content_path=str(content_path),
                vis_id="codebert",
                data_type="generic_dual_encoder",
                task_type="semantic_diagnostic",
                vis_config={
                    "D": int(vectors.shape[1]),
                    "d": 2,
                    "bs": 64,
                    "epochs_ae": 6,
                    "epochs_joint": 6,
                    "warmup_epochs": 3,
                    "lambda_dir": 2.0,
                    "lambda_rank": 0.5,
                    "norm_mode": "center_only",
                    "std_clip_low": 1.0,
                    "gpu_id": -1,
                },
            )
            runner.run()
    points = np.load(projection_path).astype(np.float32)
    if len(points) != len(nodes):
        raise ValueError("CodeBERT DynaVis projection/node count mismatch")
    minimum = points.min(axis=0)
    span = np.maximum(points.max(axis=0) - minimum, 1e-6)
    scaled = (points - minimum) / span
    scaled[:, 0] = scaled[:, 0] * 760 + 60
    scaled[:, 1] = scaled[:, 1] * 460 + 40
    return scaled


@lru_cache(maxsize=256)
def _baseline_candidate_representations(test_id: str, candidate_id: str) -> tuple[dict[str, Any], list[str], np.ndarray, list[str], np.ndarray]:
    session = build_session_payload(test_id)
    query_tokens = list(session["query"]["tokens"])
    code_index = int(candidate_id.replace("code_", ""))
    row = load_csn_codebase()[code_index]
    code_tokens = _code_tokens(row)
    (_query_pool, query_hidden), (_code_pool, code_hidden) = _encode_token_sequences([query_tokens, code_tokens])
    return row, query_tokens, query_hidden, code_tokens, code_hidden


def _candidate_representations(test_id: str, candidate_id: str) -> tuple[dict[str, Any], list[str], np.ndarray, list[str], np.ndarray]:
    row, query_tokens, query_hidden, code_tokens, code_hidden = _baseline_candidate_representations(test_id, candidate_id)
    edited_code_hidden = code_hidden.copy()
    with _LOCAL_INTERVENTION_LOCK:
        overrides = _LOCAL_CODE_OVERRIDES.get((str(test_id), candidate_id), {})
        for token_index, vector in overrides.items():
            if 0 <= token_index < len(edited_code_hidden):
                edited_code_hidden[token_index] = vector
    return row, query_tokens, query_hidden, code_tokens, edited_code_hidden


def _line_similarity_scores(concepts: list[dict[str, Any]], query_hidden: np.ndarray, code_hidden: np.ndarray, code_lines: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    normalized_code = code_hidden / np.maximum(np.linalg.norm(code_hidden, axis=1, keepdims=True), 1e-8)
    scores: dict[tuple[int, int], float] = {}
    for concept in concepts:
        indices = [index for index in concept["tokenIndices"] if index < len(query_hidden)]
        if not indices:
            continue
        vector = _normalize(query_hidden[indices].mean(axis=0))
        for line in code_lines:
            token_indices = [int(index) for index in line.get("tokenIndices", []) if int(index) < len(normalized_code)]
            if not token_indices:
                continue
            similarities = normalized_code[token_indices] @ vector
            attention = np.exp((similarities - similarities.max()) / 0.15)
            attention /= max(float(attention.sum()), 1e-8)
            pooled = _normalize((normalized_code[token_indices] * attention[:, None]).sum(axis=0))
            scores[(int(concept["conceptId"]), int(line["lineNumber"]))] = float(vector @ pooled)
    return scores


def _concept_for_query_token(concepts: list[dict[str, Any]], query_hidden: np.ndarray, query_token_index: int) -> np.ndarray:
    matching = [concept for concept in concepts if query_token_index in concept.get("tokenIndices", [])]
    indices = matching[0]["tokenIndices"] if matching else [query_token_index]
    indices = [index for index in indices if 0 <= index < len(query_hidden)]
    if not indices:
        raise ValueError("Query token has no valid CodeBERT hidden state.")
    return _normalize(query_hidden[indices].mean(axis=0))


def apply_local_intervention(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply a candidate-local residual without modifying frozen CodeBERT parameters."""
    test_id = str(payload.get("testId") or "")
    candidate_id = str(payload.get("candidateId") or "")
    if not is_supported_test(test_id) or not candidate_id.startswith("code_"):
        raise ValueError("CodeBERT local intervention requires a CSN test and CodeBERT candidate.")
    pairs = list(payload.get("pairInterventions") or [])
    if not pairs:
        return {"status": "ok", "candidates": build_session_payload(test_id)["candidates"], "localMatches": {"tokenMatches": [], "lineMatches": []}, "diagnostic": {"source": "codebert_local_semantic_what_if", "rankingChanges": False}}

    session = build_session_payload(test_id)
    concepts = session["query"]["concepts"]
    row, _query_tokens_value, query_hidden, code_tokens, baseline_code_hidden = _baseline_candidate_representations(test_id, candidate_id)
    code_lines = build_code_lines(str(row.get("code") or row.get("original_string") or ""), code_tokens)
    before_lines = _line_similarity_scores(concepts, query_hidden, baseline_code_hidden, code_lines)
    key = (test_id, candidate_id)
    changed: list[tuple[int, int, float, float]] = []
    with _LOCAL_INTERVENTION_LOCK:
        current = _LOCAL_CODE_OVERRIDES.setdefault(key, {})
        for pair in pairs:
            query_index = int(pair.get("queryTokenIndex", -1))
            code_index = int(pair.get("codeTokenIndex", -1))
            if not (0 <= code_index < len(baseline_code_hidden)) or not re.search(r"[A-Za-z_]", code_tokens[code_index]):
                continue
            concept_vector = _concept_for_query_token(concepts, query_hidden, query_index)
            original = current.get(code_index, baseline_code_hidden[code_index]).copy()
            original = _normalize(original)
            residual = concept_vector - float(original @ concept_vector) * original
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm < 1e-8:
                continue
            direction = 1.0 if float(pair.get("proximityDelta", 0.0)) >= 0 else -1.0
            strength = min(0.35, 0.08 + abs(float(pair.get("proximityDelta", 0.0))) * 0.7)
            updated = _normalize(original + direction * strength * (residual / residual_norm))
            current[code_index] = updated.astype(np.float32)
            changed.append((query_index, code_index, float(_normalize(baseline_code_hidden[code_index]) @ concept_vector), float(updated @ concept_vector)))
        _LOCAL_INTERVENTION_REVISIONS[key] = _LOCAL_INTERVENTION_REVISIONS.get(key, 0) + 1

    # Propagate the same local correction to semantically similar code tokens in
    # the loaded candidate set. This is an adapter-level counterfactual, not a
    # mutation of frozen CodeBERT parameters.
    propagated: dict[str, list[tuple[int, int, int, float, float]]] = {}
    with _LOCAL_INTERVENTION_LOCK:
        for query_index, source_index, _before, _after in changed:
            source_vector = _normalize(baseline_code_hidden[source_index])
            concept_vector = _concept_for_query_token(concepts, query_hidden, query_index)
            for target in session["candidates"]:
                target_id = str(target["id"])
                target_row, _target_query, _target_hidden, target_tokens, target_base = _baseline_candidate_representations(test_id, target_id)
                target_key = (test_id, target_id)
                target_overrides = _LOCAL_CODE_OVERRIDES.setdefault(target_key, {})
                for target_index, token in enumerate(target_tokens):
                    if target_id == candidate_id and target_index == source_index:
                        continue
                    if not re.search(r"[A-Za-z_]", token):
                        continue
                    original = _normalize(target_overrides.get(target_index, target_base[target_index]))
                    gate = float(original @ source_vector)
                    if gate < 0.68:
                        continue
                    residual = concept_vector - float(original @ concept_vector) * original
                    residual_norm = float(np.linalg.norm(residual))
                    if residual_norm < 1e-8:
                        continue
                    updated = _normalize(original + min(0.14, 0.22 * gate) * (residual / residual_norm))
                    target_overrides[target_index] = updated.astype(np.float32)
                    _LOCAL_INTERVENTION_REVISIONS[target_key] = _LOCAL_INTERVENTION_REVISIONS.get(target_key, 0) + 1
                    propagated.setdefault(target_id, []).append((query_index, target_index, next((concept["conceptId"] for concept in concepts if query_index in concept["tokenIndices"]), -1), float(_normalize(target_base[target_index]) @ concept_vector), float(updated @ concept_vector)))

    _row, _qt, _qh, _ct, edited_code_hidden = _candidate_representations(test_id, candidate_id)
    after_lines = _line_similarity_scores(concepts, query_hidden, edited_code_hidden, code_lines)
    build_candidate_payload.cache_clear()
    build_graph.cache_clear()
    token_matches = [
        {"queryTokenIndex": query_index, "codeTokenIndex": code_index, "queryToken": session["query"]["tokens"][query_index], "codeToken": code_tokens[code_index], "baseline": round(before, 6), "similarity": round(after, 6), "delta": round(after - before, 6), "color": next((concept["color"] for concept in concepts if query_index in concept["tokenIndices"]), "#8b5cf6"), "newlyConnected": False, "source": "local_drag", "role": "dragged"}
        for query_index, code_index, before, after in changed
    ]
    line_matches = [
        {"conceptId": concept_id, "lineNumber": line_number, "baseline": round(before, 6), "similarity": round(after_lines[(concept_id, line_number)], 6), "delta": round(after_lines[(concept_id, line_number)] - before, 6), "color": next((concept["color"] for concept in concepts if concept["conceptId"] == concept_id), "#8b5cf6"), "source": "local_drag"}
        for (concept_id, line_number), before in before_lines.items()
        if abs(after_lines[(concept_id, line_number)] - before) >= 1e-5
    ]
    line_matches.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    generalized_details: dict[str, Any] = {}
    for target_id, deltas in propagated.items():
        target_row, _target_query, _target_hidden, target_tokens, target_base = _baseline_candidate_representations(test_id, target_id)
        target_lines = build_code_lines(str(target_row.get("code") or target_row.get("original_string") or ""), target_tokens)
        target_before_lines = _line_similarity_scores(concepts, query_hidden, target_base, target_lines)
        _tr, _tq, _th, _tt, target_edited = _candidate_representations(test_id, target_id)
        target_after_lines = _line_similarity_scores(concepts, query_hidden, target_edited, target_lines)
        generalized_details[target_id] = {
            "originalMatches": [{"conceptId": concept_id, "lineNumber": line_number, "similarity": score} for (concept_id, line_number), score in target_before_lines.items()],
            "matches": [{"conceptId": concept_id, "lineNumber": line_number, "similarity": score} for (concept_id, line_number), score in target_after_lines.items()],
            "tokenPairDeltas": [
                {"conceptId": concept_id, "queryTokenIndex": query_index, "queryToken": session["query"]["tokens"][query_index], "codeTokenIndex": token_index, "codeToken": target_tokens[token_index], "originalSimilarity": before, "generalizedSimilarity": after, "delta": after - before}
                for query_index, token_index, concept_id, before, after in deltas
            ],
        }

    vectors, code_indices, _urls = _index_metadata()
    query_vector = _normalize(query_hidden.mean(axis=0))
    updated_scores = np.asarray(vectors @ query_vector, dtype=np.float32).copy()
    edited_ids = set(propagated) | {candidate_id}
    for target_id in edited_ids:
        _tr, _tq, _th, _tt, target_hidden = _candidate_representations(test_id, target_id)
        code_index = int(target_id.replace("code_", ""))
        matches = np.flatnonzero(code_indices == code_index)
        if len(matches):
            updated_scores[int(matches[0])] = float(_normalize(target_hidden.mean(axis=0)) @ query_vector)
    rank_order = np.argsort(updated_scores)[::-1]
    rank_by_code = {int(code_indices[position]): rank + 1 for rank, position in enumerate(rank_order)}
    baseline_rank_by_id = {str(item["id"]): int(item["rank"]) for item in session["candidates"]}
    reranked = []
    for item in session["candidates"]:
        code_index = int(item["codeIdx"])
        index_match = np.flatnonzero(code_indices == code_index)
        new_score = float(updated_scores[int(index_match[0])]) if len(index_match) else float(item["similarity"])
        new_rank = rank_by_code.get(code_index, int(item["rank"]))
        reranked.append({**item, "rank": new_rank, "originalRank": baseline_rank_by_id[str(item["id"])], "rankDelta": baseline_rank_by_id[str(item["id"])] - new_rank, "originalSimilarity": float(item["similarity"]), "similarity": round(new_score, 6), "similarityDelta": round(new_score - float(item["similarity"]), 6), "generalizedDelta": round(new_score - float(item["similarity"]), 6), "generalizationSource": "codebert_contextual_token_residual_counterfactual"})
    reranked.sort(key=lambda item: int(item["rank"]))
    return {
        "status": "ok",
        "candidates": reranked,
        "generalizedMatchesByCandidate": generalized_details,
        "localMatches": {"tokenMatches": token_matches, "lineMatches": line_matches[:8]},
        "diagnostic": {"source": "codebert_contextual_token_residual_counterfactual", "rankingChanges": True, "affectedCandidates": len(edited_ids), "message": "Counterfactual ranks use edited contextual token states re-aggregated into the adapter retrieval vector; frozen CodeBERT parameters are unchanged."},
    }


def reset_local_interventions(test_id: str | None = None) -> dict[str, Any]:
    with _LOCAL_INTERVENTION_LOCK:
        keys = [key for key in _LOCAL_CODE_OVERRIDES if test_id is None or key[0] == str(test_id)]
        for key in keys:
            _LOCAL_CODE_OVERRIDES.pop(key, None)
            _LOCAL_INTERVENTION_REVISIONS.pop(key, None)
    build_candidate_payload.cache_clear()
    build_graph.cache_clear()
    return {"status": "ok", "clearedMemories": len(keys)}


def _concept_matches(concepts: list[dict[str, Any]], query_hidden: np.ndarray, code_hidden: np.ndarray, code_lines: list[dict[str, Any]], code_tokens: list[str]) -> list[dict[str, Any]]:
    if not len(query_hidden) or not len(code_hidden):
        return []
    code_normalized = code_hidden / np.maximum(np.linalg.norm(code_hidden, axis=1, keepdims=True), 1e-8)
    line_for_token = _line_by_token(code_lines)
    matches: list[dict[str, Any]] = []
    for concept in concepts:
        indices = [index for index in concept["tokenIndices"] if index < len(query_hidden)]
        if not indices:
            continue
        vector = _normalize(query_hidden[indices].mean(axis=0))
        similarity = code_normalized @ vector
        semantic_indices = [index for index, token in enumerate(code_tokens) if re.search(r"[A-Za-z_]", token)] or list(range(len(code_tokens)))
        best = sorted(semantic_indices, key=lambda index: float(similarity[index]), reverse=True)[:4]
        if not best:
            continue
        best_line = line_for_token.get(best[0])
        matches.append({
            "id": f"match_{concept['conceptId']}",
            "conceptId": concept["conceptId"],
            "queryTokenIndices": indices,
            "codeTokenIndices": best,
            "lineNumber": best_line,
            "queryText": concept["text"],
            "codeText": " ".join(code_tokens[index] for index in best),
            "similarity": round(float(similarity[best[0]]), 6),
            "color": concept["color"],
        })
    return matches


@lru_cache(maxsize=256)
def build_candidate_payload(test_id: str, candidate_id: str) -> dict[str, Any]:
    session = build_session_payload(test_id)
    row, query_tokens, query_hidden, code_tokens, code_hidden = _candidate_representations(test_id, candidate_id)
    raw_code = str(row.get("code") or row.get("original_string") or "")
    code_lines = build_code_lines(raw_code, code_tokens)
    summary = next((item for item in session["candidates"] if item["id"] == candidate_id), None)
    has_override = bool(_LOCAL_CODE_OVERRIDES.get((str(test_id), candidate_id)))
    score = float(_normalize(query_hidden.mean(axis=0)) @ _normalize(code_hidden.mean(axis=0))) if has_override or not summary else float(summary["similarity"])
    return {
        "id": candidate_id,
        "testId": str(test_id),
        "codeIdx": int(candidate_id.replace("code_", "")),
        "queryTokens": query_tokens,
        "rawCode": raw_code,
        "codeTokens": code_tokens,
        "codeLines": code_lines,
        "similarity": round(score, 6),
        "conceptMatches": _concept_matches(session["query"]["concepts"], query_hidden, code_hidden, code_lines, code_tokens),
        "metadata": {"repo": str(row.get("repo") or ""), "path": str(row.get("path") or ""), "funcName": str(row.get("func_name") or ""), "url": str(row.get("url") or "")},
        "rankingSource": "codebert_epoch2_full_csn_dual_encoder_index",
        "conceptSource": "representation_level_semantic_alignment",
        **model_metadata(),
    }


def _hierarchy(candidate: dict[str, Any], query_hidden: np.ndarray, code_hidden: np.ndarray) -> dict[str, Any]:
    concepts = build_session_payload(candidate["testId"])["query"]["concepts"]
    code_lines = candidate["codeLines"]
    code_tokens = candidate["codeTokens"]
    blocks = _semantic_ast_blocks(candidate["rawCode"], code_lines)
    normalized_code = code_hidden / np.maximum(np.linalg.norm(code_hidden, axis=1, keepdims=True), 1e-8)
    concept_vectors = {concept["conceptId"]: _normalize(query_hidden[concept["tokenIndices"]].mean(axis=0)) for concept in concepts if concept["tokenIndices"]}
    query_nodes = [{"id": f"q_concept_{concept['conceptId']}", "type": "query_concept", "label": concept["text"], "conceptId": concept["conceptId"], "similarity": 0.0} for concept in concepts]
    line_by_number = {int(line["lineNumber"]): line for line in code_lines}

    def score_tokens(token_indices: list[int]) -> list[dict[str, Any]]:
        values = []
        for concept in concepts:
            vector = concept_vectors[concept["conceptId"]]
            valid = [index for index in token_indices if index < len(normalized_code)]
            if not valid:
                continue
            similarity = normalized_code[valid] @ vector
            attention = np.exp((similarity - similarity.max()) / 0.15)
            attention /= np.maximum(attention.sum(), 1e-8)
            pooled = _normalize((normalized_code[valid] * attention[:, None]).sum(axis=0))
            values.append({"conceptId": concept["conceptId"], "similarity": round(float(vector @ pooled), 6)})
        return sorted(values, key=lambda item: item["similarity"], reverse=True)

    block_items: list[dict[str, Any]] = []
    lines_by_block: dict[str, list[dict[str, Any]]] = {}
    block_vectors: list[np.ndarray] = []
    for block in blocks:
        block_lines = [line for line in code_lines if block["startLine"] <= int(line["lineNumber"]) <= block["endLine"]]
        token_indices = [int(index) for line in block_lines for index in line.get("tokenIndices", [])]
        if not token_indices:
            continue
        scores = score_tokens(token_indices)
        block_vector = _normalize(normalized_code[token_indices].mean(axis=0))
        block_vectors.append(block_vector)
        block_items.append({**block, "tokenIndices": token_indices, "lineNumbers": [int(line["lineNumber"]) for line in block_lines], "conceptScores": scores, "similarity": scores[0]["similarity"] if scores else 0.0, "conceptId": scores[0]["conceptId"] if scores else None, "signals": [], "label": f"{block['kind']} · L{block['startLine']}"})
        lines_by_block[block["id"]] = [
            {"lineNumber": int(line["lineNumber"]), "text": line["text"], "tokenIndices": list(line.get("tokenIndices", [])), "conceptScores": score_tokens(list(line.get("tokenIndices", []))), "similarity": (score_tokens(list(line.get("tokenIndices", []))) or [{"similarity": 0.0}])[0]["similarity"], "conceptId": (score_tokens(list(line.get("tokenIndices", []))) or [{"conceptId": None}])[0].get("conceptId"), "signals": []}
            for line in block_lines if line.get("tokenIndices")
        ]
    query_vectors = [concept_vectors[concept["conceptId"]] for concept in concepts]
    positions = _project_vectors(query_vectors + block_vectors)
    for index, node in enumerate(query_nodes): node.update(positions[index])
    for index, block in enumerate(block_items): block.update(positions[len(query_nodes) + index])
    query_nodes_by_block: dict[str, list[dict[str, Any]]] = {}
    for block in block_items:
        lines = lines_by_block[block["id"]]
        line_vectors = [_normalize(normalized_code[line["tokenIndices"]].mean(axis=0)) for line in lines]
        line_positions = _project_vectors(query_vectors + line_vectors)
        query_nodes_by_block[block["id"]] = [{**node, **line_positions[index]} for index, node in enumerate(query_nodes)]
        for index, line in enumerate(lines): line.update(line_positions[len(query_nodes) + index])
    return {"queryNodes": query_nodes, "queryNodesByBlock": query_nodes_by_block, "blocks": block_items, "linesByBlock": lines_by_block, "recommendedTokenIndices": [], "recommendedTokens": []}


@lru_cache(maxsize=256)
def build_graph(test_id: str, candidate_id: str) -> dict[str, Any]:
    candidate = build_candidate_payload(test_id, candidate_id)
    _row, query_tokens, query_hidden, code_tokens, code_hidden = _candidate_representations(test_id, candidate_id)
    concepts = build_session_payload(test_id)["query"]["concepts"]
    matches = candidate["conceptMatches"]
    query_by_token: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(query_tokens))}
    code_by_token: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(code_tokens))}
    for match in matches:
        for index in match["queryTokenIndices"]: query_by_token.setdefault(index, []).append(match)
        for index in match["codeTokenIndices"]: code_by_token.setdefault(index, []).append(match)
    vectors = np.concatenate([query_hidden, code_hidden], axis=0)
    code_line_map = _line_by_token(candidate["codeLines"])
    nodes = []
    for index, token in enumerate(query_tokens):
        items = query_by_token.get(index, [])
        primary = items[0] if items else None
        nodes.append({"id": f"q_tok_{index}", "type": "query_token", "label": token, "tokenIndex": index, "conceptId": primary["conceptId"] if primary else None, "conceptIds": [item["conceptId"] for item in items], "color": primary["color"] if primary else "#9ca3af", "colors": [item["color"] for item in items], "highlightScore": float(max((item["similarity"] for item in items), default=0.0))})
    for index, token in enumerate(code_tokens):
        items = code_by_token.get(index, [])
        primary = items[0] if items else None
        nodes.append({"id": f"c_tok_{index}", "type": "code_token", "label": token, "tokenIndex": index, "lineNumber": code_line_map.get(index), "conceptId": primary["conceptId"] if primary else None, "conceptIds": [item["conceptId"] for item in items], "color": primary["color"] if primary else "#6b7280", "colors": [item["color"] for item in items], "highlightScore": float(max((item["similarity"] for item in items), default=0.0))})
    positions = _dynavis_projection(test_id, candidate_id, vectors, nodes)
    for node, point in zip(nodes, positions):
        node.update({"x": float(point[0]), "y": float(point[1])})
    semantic_links = []
    for concept in concepts:
        vector = _normalize(query_hidden[concept["tokenIndices"]].mean(axis=0))
        similarities = (code_hidden / np.maximum(np.linalg.norm(code_hidden, axis=1, keepdims=True), 1e-8)) @ vector
        for index in np.argsort(similarities)[-5:][::-1]:
            for query_index in concept["tokenIndices"]:
                semantic_links.append({"source": f"q_tok_{query_index}", "target": f"c_tok_{int(index)}", "similarity": round(float(similarities[index]), 6), "sourceType": "query_token", "targetType": "code_token"})
    return {"testId": str(test_id), "candidateId": candidate_id, "method": "DynaVis", "epoch": 4, "availableEpochs": [1, 2, 3, 4], "representationSource": "codebert_epoch2_contextual_token_hidden_states", "representationKind": "codebert_contextual_token_hidden_states_dynavis", "queryRepresentationSource": "semantic_span_mean_pooling", "nodes": nodes, "edges": [], "semanticLinks": semantic_links, "semanticLinkSource": "codebert_contextual_cosine", "semanticLinkThreshold": 0.25, "hierarchy": _hierarchy(candidate, query_hidden, code_hidden), **model_metadata()}
