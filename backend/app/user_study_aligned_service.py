from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .aligned_xsearch_service import QueryConcept, _extract_query_concepts, _extract_query_encoding, _override_query_concepts, _query_text, model_context
from .config import CONCEPT_COLORS, USER_STUDY_ROLES_PATH, XSEARCH_ROOT
from .data_service import (
    CSN_CODE_OFFSET,
    CSN_RERANK_DEMO_CONFIG,
    build_code_lines,
    csn_actual_code_idx,
    get_row,
    is_csn_code_idx,
    is_csn_demo_test,
    load_csn_queries,
    token_text,
)


@lru_cache(maxsize=1)
def _roles() -> torch.Tensor:
    arr = np.load(USER_STUDY_ROLES_PATH, mmap_mode="r")
    return torch.from_numpy(np.asarray(arr[:, :256]).copy()).long()


def _feature_to_batch(feature: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    code_length = 256
    attn_mask = np.zeros((code_length, code_length), dtype=bool)
    node_index = sum([i > 1 for i in feature.position_idx])
    max_length = sum([i != 1 for i in feature.position_idx])
    attn_mask[:node_index, :node_index] = True
    for idx, token_id in enumerate(feature.code_ids):
        if token_id in [0, 2]:
            attn_mask[idx, :max_length] = True
    for idx, (a, b) in enumerate(feature.dfg_to_code):
        if a < node_index and b < node_index and idx + node_index < code_length:
            attn_mask[idx + node_index, a:b] = True
            attn_mask[a:b, idx + node_index] = True
    for idx, nodes in enumerate(feature.dfg_to_dfg):
        for node in nodes:
            if node + node_index < len(feature.position_idx) and idx + node_index < code_length:
                attn_mask[idx + node_index, node + node_index] = True
    return (
        torch.tensor(feature.code_ids).unsqueeze(0),
        torch.tensor(attn_mask).unsqueeze(0),
        torch.tensor(feature.position_idx).unsqueeze(0),
    )


@lru_cache(maxsize=64)
def _encode_code_row(code_idx: int) -> tuple[Any, torch.Tensor, torch.Tensor]:
    if str(XSEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(XSEARCH_ROOT))
    from dataloader import TextDataset

    tokenizer, model = model_context()
    args = SimpleNamespace(nl_length=128, code_length=256, data_flow_length=0, lang="python", device="cpu")
    row = get_row(code_idx)
    old_cwd = Path.cwd()
    os.chdir(XSEARCH_ROOT)
    try:
        feature = TextDataset.convert_examples_to_features((row, tokenizer, args), compute_alignment=False)
    finally:
        os.chdir(old_cwd)
    code_inputs, attn_mask, position_idx = _feature_to_batch(feature)
    with torch.inference_mode():
        out = model(
            code_inputs=code_inputs,
            attn_mask=attn_mask,
            position_idx=position_idx,
            role_indices=_roles()[code_idx : code_idx + 1],
        )
    hidden = torch.nan_to_num(out.code_hidden[0].detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
    scores = torch.nan_to_num(out.code_scores[0].detach().cpu().float(), nan=0.0, posinf=0.0, neginf=0.0)
    return feature, hidden, scores


def _line_centroids_from_feature(row: dict[str, Any], feature: Any, hidden: torch.Tensor, scores: torch.Tensor):
    code_tokens = list(row.get("code_tokens") or [])
    raw_code = row.get("clean_code") or row.get("code") or row.get("original_string") or ""
    lines = build_code_lines(raw_code, code_tokens)
    clusters = []
    for line in lines:
        original_indices = [int(idx) for idx in line.get("tokenIndices", []) if 0 <= int(idx) < len(code_tokens)]
        slots: list[int] = []
        for idx in original_indices:
            span = feature.ori2cur_pos.get(idx)
            if not span:
                continue
            start, end = int(span[0]) + 1, int(span[1]) + 1
            slots.extend(slot for slot in range(start, end) if 0 <= slot < hidden.shape[0])
        if not slots:
            continue
        highlighted = [slot for slot in slots if float(scores[slot].item()) > 0.0]
        used_slots = highlighted or slots
        centroid = F.normalize(hidden[used_slots].mean(dim=0), dim=0)
        centroid = torch.nan_to_num(centroid, nan=0.0, posinf=0.0, neginf=0.0)
        clusters.append(
            {
                "lineNumber": int(line["lineNumber"]),
                "tokenIndices": original_indices,
                "slots": used_slots,
                "centroid": centroid,
            }
        )
    return clusters


def _score_lines(query_concepts: list[QueryConcept], clusters: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    if not query_concepts or not clusters:
        return 0.0, []
    weights = torch.tensor([max(0.0, concept.weight) for concept in query_concepts], dtype=torch.float32)
    weights = weights / weights.sum() if float(weights.sum().item()) > 0 else torch.ones(len(weights)) / len(weights)
    code_centroids = torch.stack([cluster["centroid"] for cluster in clusters])
    matches = []
    sims_for_score = []
    for concept_id, concept in enumerate(query_concepts):
        sims = torch.matmul(code_centroids, concept.centroid)
        best_sim, best_idx = torch.max(sims, dim=0)
        cluster = clusters[int(best_idx.item())]
        sims_for_score.append(best_sim)
        matches.append(
            {
                "conceptId": concept_id,
                "queryTokenIndices": concept.indices,
                "codeTokenIndices": cluster["tokenIndices"],
                "lineNumber": cluster["lineNumber"],
                "similarity": round(float(best_sim.item()), 6),
            }
        )
    score = float((torch.stack(sims_for_score) * weights).sum().item())
    return round(score, 6), matches


def model_token_similarity(test_id: str, code_idx: int, query_token_index: int, code_token_index: int) -> dict[str, float]:
    if is_csn_demo_test(test_id):
        query_row = load_csn_queries()[int(CSN_RERANK_DEMO_CONFIG[str(test_id)]["queryIndex"])]
    else:
        query_row = get_row(int(test_id))
    query_text = _query_text(query_row)
    from .aligned_xsearch_service import _extract_query_encoding

    query_encoding = _extract_query_encoding(query_text)
    if not (0 <= query_token_index < len(query_encoding.tokens)):
        raise ValueError("Manual link query token index is out of range for model encoding.")

    row = get_row(int(code_idx))
    code_tokens = list(row.get("code_tokens") or [])
    if not (0 <= code_token_index < len(code_tokens)):
        raise ValueError("Manual link code token index is out of range for model encoding.")

    query_vec = F.normalize(query_encoding.vectors[int(query_token_index)].float(), dim=0)
    if is_csn_code_idx(int(code_idx)):
        code_vec = code_token_representations(int(code_idx)).get(int(code_token_index))
        if code_vec is None:
            raise ValueError("Manual link code token has no valid model hidden slot.")
        cosine = float(torch.dot(query_vec, code_vec).item())
        positive = max(0.0, min(1.0, cosine))
        return {
            "cosine": round(cosine, 6),
            "positive": round(positive, 6),
        }

    feature, hidden, _scores = _encode_code_row(int(code_idx))
    span = feature.ori2cur_pos.get(int(code_token_index))
    if not span:
        raise ValueError("Manual link code token is not mapped to model subtokens.")
    start, end = int(span[0]) + 1, int(span[1]) + 1
    slots = [slot for slot in range(start, end) if 0 <= slot < hidden.shape[0]]
    if not slots:
        raise ValueError("Manual link code token has no valid model hidden slot.")

    code_vec = F.normalize(hidden[slots].mean(dim=0).float(), dim=0)
    cosine = float(torch.dot(query_vec, code_vec).item())
    positive = max(0.0, min(1.0, cosine))
    return {
        "cosine": round(cosine, 6),
        "positive": round(positive, 6),
    }


def query_representation_context(test_id: str) -> tuple[list[str], torch.Tensor, list[QueryConcept]]:
    if is_csn_demo_test(test_id):
        query_row = load_csn_queries()[int(CSN_RERANK_DEMO_CONFIG[str(test_id)]["queryIndex"])]
    else:
        query_row = get_row(int(test_id))
    encoding = _extract_query_encoding(_query_text(query_row))
    vectors = F.normalize(encoding.vectors.detach().cpu().float(), dim=1)
    return encoding.tokens, vectors, _override_query_concepts(_query_text(query_row), vectors, encoding.concepts)


def code_token_representation(code_idx: int, code_token_index: int) -> torch.Tensor:
    vectors = code_token_representations(int(code_idx))
    try:
        return vectors[int(code_token_index)]
    except KeyError as exc:
        raise ValueError("Code token has no valid model hidden slot.") from exc


@lru_cache(maxsize=128)
def code_token_representations(code_idx: int) -> dict[int, torch.Tensor]:
    if is_csn_code_idx(int(code_idx)):
        from .aligned_xsearch_service import _code_vectors_for_url, _subset_code_token_slots

        row = get_row(int(code_idx))
        code_pack = _code_vectors_for_url(row.get("url", ""))
        if code_pack is None:
            return {}
        hidden, _scores = code_pack
        code_tokens = list(row.get("code_tokens") or [])
        vectors: dict[int, torch.Tensor] = {}
        for code_token_index in range(len(code_tokens)):
            slots = _subset_code_token_slots(str(row.get("url") or ""), code_token_index)
            if slots is None:
                slots = [code_token_index + 1]
            slots = [slot for slot in slots if 0 <= slot < hidden.shape[0]]
            if not slots:
                continue
            vectors[int(code_token_index)] = F.normalize(hidden[slots].mean(dim=0).float(), dim=0)
        return vectors

    feature, hidden, _scores = _encode_code_row(int(code_idx))
    row = get_row(int(code_idx))
    code_tokens = list(row.get("code_tokens") or [])
    vectors: dict[int, torch.Tensor] = {}
    for code_token_index in range(len(code_tokens)):
        span = feature.ori2cur_pos.get(int(code_token_index))
        if not span:
            continue
        start, end = int(span[0]) + 1, int(span[1]) + 1
        slots = [slot for slot in range(start, end) if 0 <= slot < hidden.shape[0]]
        if not slots:
            continue
        vectors[int(code_token_index)] = F.normalize(hidden[slots].mean(dim=0).float(), dim=0)
    return vectors


@lru_cache(maxsize=128)
def candidate_representation_context(code_idx: int) -> list[dict[str, Any]]:
    if is_csn_code_idx(int(code_idx)):
        from .aligned_xsearch_service import _code_vectors_for_url, _line_centroids

        row = get_row(int(code_idx))
        code_pack = _code_vectors_for_url(row.get("url", ""))
        if code_pack is None:
            return []
        hidden, scores = code_pack
        return _line_centroids(row, hidden, scores)

    row = get_row(int(code_idx))
    feature, hidden, scores = _encode_code_row(int(code_idx))
    return _line_centroids_from_feature(row, feature, hidden, scores)


def score_candidate_representation(
    query_concepts: list[QueryConcept],
    query_vectors: torch.Tensor,
    code_idx: int,
    code_clusters: list[dict[str, Any]] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    corrected_concepts = []
    for concept in query_concepts:
        valid = [idx for idx in concept.indices if 0 <= int(idx) < query_vectors.shape[0]]
        if not valid:
            continue
        corrected_concepts.append(
            QueryConcept(
                indices=valid,
                centroid=F.normalize(query_vectors[valid].mean(dim=0), dim=0),
                weight=concept.weight,
            )
        )
    return _score_lines(corrected_concepts, code_clusters if code_clusters is not None else candidate_representation_context(int(code_idx)))


def build_user_study_aligned_candidate(test_id: str, candidate_id: str, ranking_score: float | None) -> dict[str, Any]:
    query_row = get_row(int(test_id))
    code_idx = int(candidate_id.replace("code_", ""))
    row = get_row(code_idx)
    query_tokens, query_concepts = _extract_query_concepts(_query_text(query_row))
    feature, hidden, scores = _encode_code_row(code_idx)
    clusters = _line_centroids_from_feature(row, feature, hidden, scores)
    computed_score, matches = _score_lines(query_concepts, clusters)

    code_tokens = list(row.get("code_tokens") or [])
    concept_matches = []
    for match in matches:
        concept_id = int(match["conceptId"])
        q_indices = [idx for idx in match["queryTokenIndices"] if 0 <= idx < len(query_tokens)]
        c_indices = [idx for idx in match["codeTokenIndices"] if 0 <= idx < len(code_tokens)]
        if not q_indices or not c_indices:
            continue
        concept_matches.append(
            {
                "id": f"match_{concept_id}",
                "conceptId": concept_id,
                "queryTokenIndices": q_indices,
                "codeTokenIndices": c_indices,
                "lineNumber": match["lineNumber"],
                "queryText": token_text(query_tokens, q_indices),
                "codeText": token_text(code_tokens, c_indices),
                "similarity": match["similarity"],
                "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
            }
        )

    raw_code = row.get("clean_code") or row.get("code") or row.get("original_string") or ""
    return {
        "id": f"code_{code_idx}",
        "testId": str(test_id),
        "codeIdx": code_idx,
        "queryTokens": query_tokens,
        "rawCode": raw_code,
        "codeTokens": code_tokens,
        "codeLines": build_code_lines(raw_code, code_tokens),
        "similarity": round(float(ranking_score if ranking_score is not None else computed_score), 6),
        "conceptMatches": concept_matches,
        "metadata": {
            "repo": row.get("repo", ""),
            "path": row.get("path", ""),
            "funcName": row.get("func_name", ""),
            "url": row.get("url", ""),
        },
        "rankingSource": "xsearch_step7000_checkpoint_user_study_recomputed",
        "conceptSource": "xsearch_step7000_query_to_code_line_recomputed",
    }
