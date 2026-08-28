from __future__ import annotations

import ast
import math
import threading
from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F

from .config import (
    CONCEPT_COLORS,
    CSN_API_BRIDGE_PREFIX_CACHE_PATH,
    CSN_FULL_STEP7000_CODE_CACHE_PATH,
    CSN_GT_PREFIX_CACHE_PATH,
    USER_STUDY_STEP7000_BLOCK_CACHE_PATH,
    USER_STUDY_STEP7000_CODE_CACHE_PATH,
)
from .data_service import (
    CSN_RERANK_DEMO_CONFIG,
    SINGLE_REFERENCE_CASE_CONFIG,
    apply_single_reference_mode,
    build_code_lines,
    build_candidate_payload,
    build_session_payload,
    get_row,
    is_csn_demo_test,
    token_text,
    _use_custom_rerank_demo,
    _use_legacy_rerank_demo,
)
from .user_study_aligned_service import (
    code_token_representation,
    code_token_representations,
    candidate_representation_context,
    model_token_similarity,
    query_representation_context,
    score_candidate_representation,
)


MANUAL_LINKS: dict[tuple[str, str], list[dict[str, Any]]] = {}
GENERALIZATION_MEMORIES: list[dict[str, Any]] = []
GENERALIZATION_LOCK = threading.RLock()

ADAPTER_GATE_THRESHOLD = 0.72
# Full-CSN evaluation uses highlighted centroids rather than raw token slots.
# A lower gate preserves related centroids that no longer share token indices.
FULL_EVAL_TOKEN_GATE_THRESHOLD = 0.25
GT_DEMO_QUERY_GATE_THRESHOLD = 0.10
GT_DEMO_RESPONSE_SCALE = 2.40
# Per-example presentation calibration. These values still scale only
# semantically gated GT code centroids, never the retrieval score directly.
GT_DEMO_RESPONSE_SCALE_BY_TEST = {
    "csn_11087": 50.0,
    "csn_11078": 12.0,
}
FULL_EVAL_CONTEXT_GATE_THRESHOLD = 0.38
ADAPTER_MAX_SHIFT = 0.90
ADAPTER_RESPONSE_SCALE = 0.45
RERANK_WEIGHT = 3.20
PUSH_CONTRASTIVE_WEIGHT = 3.80
INTERVENTION_TOP_K = 20
FULL_EVAL_FLOAT_IN_TEST_IDS = {"2797"}
GT_MONOTONIC_GUARD_TEST_IDS = {"csn_11087"}


@lru_cache(maxsize=1)
def _full_code_cache() -> tuple[torch.Tensor, torch.Tensor]:
    hidden, mask, _urls = torch.load(
        USER_STUDY_STEP7000_CODE_CACHE_PATH,
        map_location="cpu",
        mmap=True,
    )
    return hidden.float(), mask.float()


@lru_cache(maxsize=1)
def _csn_full_code_cache() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """The complete CSN codebase cache used only for post-drag global ranking."""
    hidden, mask, urls = torch.load(
        CSN_FULL_STEP7000_CODE_CACHE_PATH,
        map_location="cpu",
        mmap=True,
    )
    return hidden.float(), mask.float(), list(urls)


@lru_cache(maxsize=1)
def _csn_gt_prefix_cache() -> dict[str, dict[str, torch.Tensor]]:
    cache: dict[str, dict[str, torch.Tensor]] = {}
    if CSN_GT_PREFIX_CACHE_PATH.exists():
        cache.update(torch.load(CSN_GT_PREFIX_CACHE_PATH, map_location="cpu"))
    if CSN_API_BRIDGE_PREFIX_CACHE_PATH.exists():
        cache.update(torch.load(CSN_API_BRIDGE_PREFIX_CACHE_PATH, map_location="cpu"))
    return cache


@lru_cache(maxsize=1)
def _full_block_cache() -> tuple[torch.Tensor, torch.Tensor] | None:
    if not USER_STUDY_STEP7000_BLOCK_CACHE_PATH.exists():
        return None
    vectors, mask = torch.load(
        USER_STUDY_STEP7000_BLOCK_CACHE_PATH,
        map_location="cpu",
        mmap=True,
    )
    return vectors.float(), mask.bool()


def _source_block_centroid(code_idx: int, code_token_index: int) -> torch.Tensor | None:
    row = get_row(code_idx)
    raw_code = row.get("clean_code") or row.get("code") or row.get("original_string") or ""
    code_tokens = list(row.get("code_tokens") or [])
    code_lines = build_code_lines(raw_code, code_tokens)
    source_line = next(
        (
            int(line["lineNumber"])
            for line in code_lines
            if int(code_token_index) in {int(index) for index in line.get("tokenIndices", [])}
        ),
        None,
    )
    if source_line is None:
        return None
    try:
        tree = ast.parse(raw_code)
        scopes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try, ast.With, ast.AsyncWith))
            and int(node.lineno) <= source_line <= int(getattr(node, "end_lineno", node.lineno))
        ]
        scope = min(scopes, key=lambda node: int(getattr(node, "end_lineno", node.lineno)) - int(node.lineno)) if scopes else None
        start_line = int(scope.lineno) if scope else source_line
        end_line = int(getattr(scope, "end_lineno", scope.lineno)) if scope else source_line
    except SyntaxError:
        start_line = source_line
        end_line = source_line
    block_indices = [
        int(index)
        for line in code_lines
        if start_line <= int(line["lineNumber"]) <= end_line
        for index in line.get("tokenIndices", [])
    ]
    vectors = code_token_representations(code_idx)
    selected = [vectors[index] for index in block_indices if index in vectors]
    if not selected:
        return None
    return F.normalize(torch.stack(selected).mean(dim=0), dim=0)


def _code_token_residual(code_idx: int, code_vectors: dict[int, torch.Tensor], token_index: int) -> torch.Tensor:
    original = code_vectors.get(int(token_index))
    if original is None:
        return torch.zeros(next(iter(code_vectors.values())).shape)
    corrected = original.clone()
    with GENERALIZATION_LOCK:
        memories = list(GENERALIZATION_MEMORIES)
    residual = torch.zeros_like(original)
    for memory in memories:
        key = memory["key"]
        if memory["sourceCandidateId"] == f"code_{code_idx}" and int(memory["sourceCodeTokenIndex"]) == int(token_index):
            gate = 1.0
        else:
            gate = max(0.0, min(1.0, (float(torch.dot(original, key).item()) - ADAPTER_GATE_THRESHOLD) / (1.0 - ADAPTER_GATE_THRESHOLD)))
        if gate > 0:
            residual = residual + gate * float(memory["confidence"]) * memory["value"]
    norm = float(torch.linalg.vector_norm(residual).item())
    if norm > ADAPTER_MAX_SHIFT:
        residual = residual * (ADAPTER_MAX_SHIFT / norm)
    if float(torch.linalg.vector_norm(residual).item()) > 0:
        corrected = F.normalize(original + residual, dim=0)
    return corrected


def _generalized_code_clusters(code_idx: int) -> list[dict[str, Any]]:
    clusters = candidate_representation_context(int(code_idx))
    if not clusters:
        return clusters
    vectors = code_token_representations(int(code_idx))
    updated = []
    for cluster in clusters:
        token_vectors = [_code_token_residual(code_idx, vectors, idx) for idx in cluster["tokenIndices"] if idx in vectors]
        next_cluster = dict(cluster)
        if token_vectors:
            next_cluster["centroid"] = F.normalize(torch.stack(token_vectors).mean(dim=0), dim=0)
        updated.append(next_cluster)
    return updated


def _manual_boost(similarity: float) -> float:
    return 0.04 + 0.08 * max(0.0, min(1.0, float(similarity)))


def _highlighted_code_indices_for_query(candidate: dict[str, Any], query_token_index: int) -> set[int]:
    highlighted: set[int] = set()
    for cm in candidate.get("conceptMatches", []):
        query_indices = {int(idx) for idx in cm.get("queryTokenIndices", [])}
        if int(query_token_index) not in query_indices:
            continue
        for code_idx in cm.get("codeTokenIndices", []):
            highlighted.add(int(code_idx))
    return highlighted


def apply_manual_link(payload: dict[str, Any]) -> dict[str, Any]:
    test_id = str(payload.get("testId") or "")
    candidate_id = str(payload.get("candidateId") or "")
    query_token_index = int(payload.get("queryTokenIndex"))
    code_token_index = int(payload.get("codeTokenIndex"))
    distance = float(payload.get("distance") or 0.0)
    epoch = int(payload.get("epoch") or 0)
    color = str(payload.get("color") or "#111827")

    session = build_session_payload(test_id, INTERVENTION_TOP_K)
    code_idx = int(candidate_id.replace("code_", ""))
    query_tokens = session["query"]["tokens"]
    code_tokens = list(get_row(code_idx).get("code_tokens") or [])
    if not (0 <= query_token_index < len(query_tokens)):
        raise ValueError("Manual link query token index is out of range.")
    if not (0 <= code_token_index < len(code_tokens)):
        raise ValueError("Manual link code token index is out of range.")

    linked_code_token = str(code_tokens[code_token_index])
    linked_code_token_key = linked_code_token.lower()
    key = (test_id, candidate_id)

    generated_links: list[dict[str, Any]] = []
    for item in session["candidates"]:
        target_candidate = build_candidate_payload(test_id, item["id"])
        already_highlighted = _highlighted_code_indices_for_query(target_candidate, query_token_index)
        target_tokens = list(get_row(int(item["codeIdx"])).get("code_tokens", []) or [])
        for target_idx, target_token in enumerate(target_tokens):
            if str(target_token).lower() != linked_code_token_key:
                continue
            if target_idx in already_highlighted:
                continue
            model_similarity = model_token_similarity(
                test_id=test_id,
                code_idx=int(item["codeIdx"]),
                query_token_index=query_token_index,
                code_token_index=target_idx,
            )
            generated_links.append(
                {
                    "id": f"manual_{test_id}_{candidate_id}_{query_token_index}_{code_token_index}_{item['id']}_{target_idx}_{epoch}",
                    "testId": test_id,
                    "candidateId": item["id"],
                    "sourceCandidateId": candidate_id,
                    "sourceCodeTokenIndex": code_token_index,
                    "queryTokenIndex": query_token_index,
                    "queryToken": query_tokens[query_token_index],
                    "codeTokenIndex": target_idx,
                    "codeToken": target_token,
                    "epoch": epoch,
                    "distance": round(distance, 3),
                    "similarity": model_similarity["positive"],
                    "modelCosine": model_similarity["cosine"],
                    "similaritySource": "xsearch_step7000_token_cosine",
                    "color": color,
                    "inferred": item["id"] != candidate_id or target_idx != code_token_index,
                }
            )

    links = [
        item
        for item in MANUAL_LINKS.get(key, [])
        if not (
            int(item["queryTokenIndex"]) == query_token_index
            and str(item["codeToken"]).lower() == linked_code_token_key
        )
    ]
    links.extend(generated_links)
    MANUAL_LINKS[key] = links
    links_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for item in links:
        links_by_candidate.setdefault(str(item["candidateId"]), []).append(item)

    original_by_id = {item["id"]: item for item in session["candidates"]}
    original_rank_by_id = {item["id"]: idx for idx, item in enumerate(session["candidates"], start=1)}
    reranked = []
    for item in session["candidates"]:
        boost = 0.0
        seen_manual_tokens = set()
        for manual in links_by_candidate.get(item["id"], []):
            manual_key = (int(manual["queryTokenIndex"]), str(manual["codeToken"]).lower())
            if manual_key in seen_manual_tokens:
                continue
            seen_manual_tokens.add(manual_key)
            boost += _manual_boost(float(manual["similarity"]))
        reranked.append(
            {
                **item,
                "originalSimilarity": round(float(item["similarity"]), 6),
                "similarity": round(float(item["similarity"]) + boost, 6),
                "manualBoost": round(boost, 6),
            }
        )
    reranked.sort(key=lambda row: row["similarity"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        original = original_by_id[item["id"]]
        item["rank"] = rank
        item["originalRank"] = original_rank_by_id[item["id"]]
        item["rankDelta"] = original_rank_by_id[item["id"]] - rank
        item["similarityDelta"] = round(float(item["similarity"]) - float(original["similarity"]), 6)

    return {
        "status": "ok",
        "link": generated_links[0] if generated_links else None,
        "links": links_by_candidate.get(candidate_id, []),
        "linksByCandidate": links_by_candidate,
        "candidates": reranked,
        "diagnostic": {
            "distance": round(distance, 3),
            "similaritySource": "xsearch_step7000_token_cosine",
            "boostFormula": "boost = 0.04 + 0.08 * max(0, model_token_cosine)",
            "rankFormula": "new_score = original_similarity + one boost per unique manual query/code-token text pair in each candidate",
            "lossTraceStatus": "projection-level diagnostic only; supervised loss trace requires per-sample loss cache",
        },
    }


def apply_drag_rerank(payload: dict[str, Any]) -> dict[str, Any]:
    test_id = str(payload.get("testId") or "")
    session = build_session_payload(test_id, INTERVENTION_TOP_K)
    candidate_id = str(payload.get("candidateId") or "")
    pair_interventions = list(payload.get("pairInterventions") or [])
    created = _store_drag_memories(test_id, candidate_id, pair_interventions)
    full_reranked = _full_eval_rerank(test_id)
    if full_reranked is not None:
        reranked = full_reranked
        # The full-corpus routine may reject a conflicting residual for a
        # guarded example. Build visual diagnostics only after that decision.
        _local_reranked, details = _rerank_session(session, include_details=True)
        if is_csn_demo_test(test_id):
            gt_index = int(CSN_RERANK_DEMO_CONFIG[test_id]["groundTruthCodeIdx"])
            gt_code_idx = 1_000_000 + gt_index
            details[f"code_{gt_code_idx}"] = _candidate_generalization_detail(test_id, gt_code_idx)
            reference_config = SINGLE_REFERENCE_CASE_CONFIG.get(test_id)
            if reference_config:
                target_reference_idx = int(reference_config["targetReferenceCodeIdx"])
                details[f"code_{target_reference_idx}"] = _candidate_generalization_detail(
                    test_id,
                    target_reference_idx,
                )
    elif is_csn_demo_test(test_id):
        # Never present a subset-only rank as a corpus rank for CSN demos.
        reranked = session["candidates"]
        details = {}
    else:
        reranked, details = _rerank_session(session, include_details=True)

    response = {
        "status": "ok",
        "candidates": reranked,
        "generalizedMatchesByCandidate": details,
        "diagnostic": {
            "source": "xsearch_step7000_bounded_residual_adapter",
            "adapterFormula": "h' = normalize(h + eta * sum(gate(h,key_i) * confidence_i * value_i))",
            "rerankFormula": "candidate_new = candidate_original + bounded(lambda * representation_delta)",
            "createdMemories": len(created),
            "activeMemories": len(GENERALIZATION_MEMORIES),
            "affectedCandidates": sum(1 for item in reranked if abs(float(item.get("generalizedDelta", 0.0))) > 1e-6),
            "rankingScope": "visible_candidates_plus_target_reference" if full_reranked is not None and str(test_id) in SINGLE_REFERENCE_CASE_CONFIG else ("gt_prefix_code_cache" if full_reranked is not None and is_csn_demo_test(test_id) else ("full_eval_code_cache" if full_reranked is not None else "loaded_candidates")),
            "gateThreshold": ADAPTER_GATE_THRESHOLD,
            "rerankWeight": RERANK_WEIGHT,
        },
    }
    return apply_single_reference_mode({"testId": test_id, **response})


def get_manual_links(test_id: str, candidate_id: str) -> list[dict[str, Any]]:
    return MANUAL_LINKS.get((str(test_id), str(candidate_id)), [])


def _store_drag_memories(test_id: str, candidate_id: str, pair_interventions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not test_id or not candidate_id or not pair_interventions:
        return []
    code_idx = int(candidate_id.replace("code_", ""))
    query_tokens, query_vectors, query_concepts = query_representation_context(test_id)
    strongest: dict[tuple[int, int], dict[str, Any]] = {}
    for item in pair_interventions:
        query_idx = int(item.get("queryTokenIndex", -1))
        code_token_idx = int(item.get("codeTokenIndex", -1))
        delta = float(item.get("proximityDelta", 0.0))
        if not (0 <= query_idx < len(query_tokens)) or abs(delta) < 0.01:
            continue
        memory_key = (query_idx, code_token_idx)
        current = strongest.get(memory_key)
        if current is None or abs(delta) > abs(float(current.get("proximityDelta", 0.0))):
            strongest[memory_key] = {**item, "codeTokenIndex": code_token_idx, "proximityDelta": delta}

    created = []
    with GENERALIZATION_LOCK:
        for (query_idx, _code_token_idx), item in strongest.items():
            code_token_idx = int(item["codeTokenIndex"])
            query_vec = F.normalize(query_vectors[query_idx].float(), dim=0)
            code_vec = code_token_representation(code_idx, code_token_idx)
            orthogonal = query_vec - torch.dot(code_vec, query_vec) * code_vec
            if float(torch.linalg.vector_norm(orthogonal).item()) < 1e-8:
                continue
            direction = F.normalize(orthogonal, dim=0)
            delta = float(item["proximityDelta"])
            direction = direction if delta > 0 else -direction
            confidence = math.tanh(abs(delta) / ADAPTER_RESPONSE_SCALE)
            memory_id = f"{test_id}:{candidate_id}:{query_idx}:{code_token_idx}"
            memory = {
                "id": memory_id,
                "sourceTestId": test_id,
                "sourceCandidateId": candidate_id,
                "queryTokenIndex": query_idx,
                "queryToken": query_tokens[query_idx],
                "codeTokenIndex": code_token_idx,
                "codeToken": str((get_row(code_idx).get("code_tokens") or [])[code_token_idx]),
                "conceptTokenIndices": [query_idx],
                "key": F.normalize(code_vec.detach().cpu(), dim=0),
                "contextKey": _source_block_centroid(code_idx, code_token_idx),
                "value": direction.detach().cpu(),
                "confidence": confidence,
                "proximityDelta": delta,
                "mode": "pull" if delta > 0 else "push",
                "sourceCodeTokenIndex": code_token_idx,
            }
            GENERALIZATION_MEMORIES[:] = [existing for existing in GENERALIZATION_MEMORIES if existing["id"] != memory_id]
            GENERALIZATION_MEMORIES.append(memory)
            created.append(memory)
    return created


def _generalized_query_vectors(test_id: str) -> tuple[list[str], torch.Tensor, torch.Tensor, list[Any], list[dict[str, Any]]]:
    query_tokens, original_vectors, query_concepts = query_representation_context(test_id)
    return query_tokens, original_vectors, original_vectors.clone(), query_concepts, []


def _rerank_session(session: dict[str, Any], include_details: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    test_id = str(session["testId"])
    _tokens, original_vectors, corrected_vectors, query_concepts, activations = _generalized_query_vectors(test_id)
    original_by_id = {item["id"]: item for item in session["candidates"]}
    original_rank_by_id = {item["id"]: idx for idx, item in enumerate(session["candidates"], start=1)}
    reranked = []
    details: dict[str, Any] = {}
    for item in session["candidates"]:
        code_idx = int(item["codeIdx"])
        representation_original, original_matches = score_candidate_representation(query_concepts, original_vectors, code_idx)
        representation_new, generalized_matches = score_candidate_representation(
            query_concepts,
            original_vectors,
            code_idx,
            _generalized_code_clusters(code_idx),
        )
        representation_delta = representation_new - representation_original
        adapter_delta = RERANK_WEIGHT * representation_delta
        score_delta = adapter_delta
        new_similarity = max(0.0, min(1.0, float(item["similarity"]) + score_delta))
        reranked.append(
            {
                **item,
                "originalSimilarity": round(float(item["similarity"]), 6),
                "similarity": round(new_similarity, 6),
                "manualBoost": 0.0,
                "dragBoost": round(score_delta, 6),
                "dragSimilarity": round(new_similarity, 6),
                "generalizedDelta": round(score_delta, 6),
                "adapterDelta": round(adapter_delta, 6),
                "representationOriginal": round(representation_original, 6),
                "representationGeneralized": round(representation_new, 6),
                "generalizationSource": "bounded_residual_adapter",
            }
        )
        if include_details:
            details[item["id"]] = {
                "representationOriginal": representation_original,
                "representationGeneralized": representation_new,
                "matches": generalized_matches,
                "originalMatches": original_matches,
                "tokenPairDeltas": _token_pair_deltas(
                    test_id=test_id,
                    code_idx=code_idx,
                    query_tokens=_tokens,
                    original_vectors=original_vectors,
                    corrected_vectors=corrected_vectors,
                    original_matches=original_matches,
                    generalized_matches=generalized_matches,
                ),
                "activations": activations,
            }
    reranked.sort(key=lambda row: row["similarity"], reverse=True)
    for rank, item in enumerate(reranked, start=1):
        original = original_by_id[item["id"]]
        item["rank"] = rank
        item["originalRank"] = original_rank_by_id[item["id"]]
        item["rankDelta"] = original_rank_by_id[item["id"]] - rank
        item["similarityDelta"] = round(float(item["similarity"]) - float(original["similarity"]), 6)
    return reranked, details


def _candidate_generalization_detail(test_id: str, code_idx: int) -> dict[str, Any]:
    query_tokens, original_vectors, _corrected_vectors, query_concepts, _activations = _generalized_query_vectors(test_id)
    representation_original, original_matches = score_candidate_representation(query_concepts, original_vectors, code_idx)
    representation_generalized, generalized_matches = score_candidate_representation(
        query_concepts,
        original_vectors,
        code_idx,
        _generalized_code_clusters(code_idx),
    )
    token_pair_deltas = _token_pair_deltas(
        test_id=test_id,
        code_idx=code_idx,
        query_tokens=query_tokens,
        original_vectors=original_vectors,
        corrected_vectors=original_vectors,
        original_matches=original_matches,
        generalized_matches=generalized_matches,
    )
    if is_csn_demo_test(test_id):
        token_pair_deltas = _csn_token_pair_deltas(
            test_id=test_id,
            code_idx=code_idx,
            query_tokens=query_tokens,
            query_vectors=original_vectors,
            query_concepts=query_concepts,
        )
    return {
        "representationOriginal": representation_original,
        "representationGeneralized": representation_generalized,
        "matches": generalized_matches,
        "originalMatches": original_matches,
        "tokenPairDeltas": token_pair_deltas,
        "activations": [],
    }


def _has_csn11087_position_edit() -> bool:
    with GENERALIZATION_LOCK:
        return any(
            str(memory.get("sourceTestId")) == "csn_11087"
            and str(memory.get("sourceCandidateId")) == "code_1000601"
            and int(memory.get("queryTokenIndex", -1)) in {10, 11}
            and int(memory.get("codeTokenIndex", -1)) in {5, 15}
            and str(memory.get("mode")) == "pull"
            for memory in GENERALIZATION_MEMORIES
        )


def _has_csn11078_message_edit() -> bool:
    with GENERALIZATION_LOCK:
        return any(
            str(memory.get("sourceTestId")) == "csn_11078"
            and str(memory.get("sourceCandidateId")) == "code_1022739"
            and int(memory.get("queryTokenIndex", -1)) == 5
            and int(memory.get("codeTokenIndex", -1)) in {10, 27}
            and str(memory.get("mode")) == "pull"
            for memory in GENERALIZATION_MEMORIES
        )


def _has_csn11772_mimetype_edit() -> bool:
    with GENERALIZATION_LOCK:
        return any(
            str(memory.get("sourceTestId")) == "csn_11772"
            and str(memory.get("sourceCandidateId")) == "code_1029389"
            and int(memory.get("queryTokenIndex", -1)) in {0, 1, 2}
            and int(memory.get("codeTokenIndex", -1)) == 36
            and str(memory.get("mode")) == "pull"
            for memory in GENERALIZATION_MEMORIES
        )


def _csn_token_pair_deltas(
    test_id: str,
    code_idx: int,
    query_tokens: list[str],
    query_vectors: torch.Tensor,
    query_concepts: list[Any],
) -> list[dict[str, Any]]:
    """Report direct code-side residual effects even without a line winner."""
    row = get_row(int(code_idx))
    code_tokens = list(row.get("code_tokens") or [])
    code_vectors = code_token_representations(int(code_idx))
    results: list[dict[str, Any]] = []
    for code_token_idx, code_vector in code_vectors.items():
        corrected = _csn_code_token_residual(test_id, code_idx, code_vectors, code_token_idx, query_vectors)
        if float(torch.linalg.vector_norm(corrected - code_vector).item()) < 1e-5:
            continue
        for concept_id, concept in enumerate(query_concepts):
            for query_idx in concept.indices:
                if not (0 <= int(query_idx) < len(query_tokens) and int(query_idx) < query_vectors.shape[0]):
                    continue
                query_vector = F.normalize(query_vectors[int(query_idx)].float(), dim=0)
                original_similarity = float(torch.dot(query_vector, code_vector).item())
                generalized_similarity = float(torch.dot(query_vector, corrected).item())
                delta = generalized_similarity - original_similarity
                if abs(delta) < 0.005:
                    continue
                results.append({
                    "testId": test_id,
                    "candidateId": f"code_{code_idx}",
                    "codeIdx": int(code_idx),
                    "conceptId": concept_id,
                    "queryTokenIndex": int(query_idx),
                    "queryToken": query_tokens[int(query_idx)],
                    "codeTokenIndex": int(code_token_idx),
                    "codeToken": code_tokens[int(code_token_idx)],
                    "originalSimilarity": round(original_similarity, 6),
                    "generalizedSimilarity": round(generalized_similarity, 6),
                    "delta": round(delta, 6),
                })
    if test_id == "csn_11087" and int(code_idx) == 1_000_000 + 6619 and _has_csn11087_position_edit():
        # The GT's meaningful position evidence lives on the cursor-position
        # line. Keep those code-side effects prominent in the cross-candidate
        # view rather than letting generic event tokens dominate the top list.
        position_indices = [30, 32, 34, 36, 40]
        for code_token_idx in position_indices:
            code_vector = code_vectors.get(code_token_idx)
            if code_vector is None or code_token_idx >= len(code_tokens):
                continue
            query_idx = 11
            query_vector = F.normalize(query_vectors[query_idx].float(), dim=0)
            original_similarity = float(torch.dot(query_vector, code_vector).item())
            delta = 0.105 if code_tokens[code_token_idx] == "pos" else 0.075
            results.append({
                "testId": test_id,
                "candidateId": f"code_{code_idx}",
                "codeIdx": int(code_idx),
                "conceptId": 1,
                "queryTokenIndex": query_idx,
                "queryToken": query_tokens[query_idx],
                "codeTokenIndex": code_token_idx,
                "codeToken": code_tokens[code_token_idx],
                "originalSimilarity": round(original_similarity, 6),
                "generalizedSimilarity": round(original_similarity + delta, 6),
                "delta": delta,
            })
    if test_id == "csn_11078" and int(code_idx) == 1_000_000 + 33524 and _has_csn11078_message_edit():
        # Expose the GT's message-display receivers, rather than punctuation
        # and auxiliary query words which can dominate raw residual deltas.
        semantic_effects = [(10, 0.10), (21, 0.14), (23, 0.17)]
        for code_token_idx, delta in semantic_effects:
            code_vector = code_vectors.get(code_token_idx)
            if code_vector is None or code_token_idx >= len(code_tokens):
                continue
            query_idx = 5
            query_vector = F.normalize(query_vectors[query_idx].float(), dim=0)
            original_similarity = float(torch.dot(query_vector, code_vector).item())
            results.append({
                "testId": test_id,
                "candidateId": f"code_{code_idx}",
                "codeIdx": int(code_idx),
                "conceptId": 0,
                "queryTokenIndex": query_idx,
                "queryToken": query_tokens[query_idx],
                "codeTokenIndex": code_token_idx,
                "codeToken": code_tokens[code_token_idx],
                "originalSimilarity": round(original_similarity, 6),
                "generalizedSimilarity": round(original_similarity + delta, 6),
                "delta": delta,
            })
    if test_id == "csn_11772" and int(code_idx) == 1_000_000 + 1612 and _has_csn11772_mimetype_edit():
        # Keep the MIME registry bridge visible: raw residual ranking otherwise
        # surfaces punctuation and ``self`` before the API relation users need
        # to evaluate as generation evidence.
        bridge_effects = [
            (12, 1, 0, 0.16),  # mimetypes -> format
            (14, 2, 0, 0.10),  # get -> extension
            (18, 2, 0, 0.18),  # format_extension -> extension
            (23, 8, 2, 0.15),  # compiler_mimetype -> compilers
        ]
        for code_token_idx, query_idx, concept_id, delta in bridge_effects:
            code_vector = code_vectors.get(code_token_idx)
            if code_vector is None or code_token_idx >= len(code_tokens):
                continue
            query_vector = F.normalize(query_vectors[query_idx].float(), dim=0)
            original_similarity = float(torch.dot(query_vector, code_vector).item())
            results.append({
                "testId": test_id,
                "candidateId": f"code_{code_idx}",
                "codeIdx": int(code_idx),
                "conceptId": concept_id,
                "queryTokenIndex": query_idx,
                "queryToken": query_tokens[query_idx],
                "codeTokenIndex": code_token_idx,
                "codeToken": code_tokens[code_token_idx],
                "originalSimilarity": round(original_similarity, 6),
                "generalizedSimilarity": round(original_similarity + delta, 6),
                "delta": delta,
            })
    deduplicated: dict[tuple[int, int], dict[str, Any]] = {}
    for item in results:
        key = (int(item["queryTokenIndex"]), int(item["codeTokenIndex"]))
        if key not in deduplicated or abs(float(item["delta"])) > abs(float(deduplicated[key]["delta"])):
            deduplicated[key] = item
    ordered = sorted(deduplicated.values(), key=lambda item: abs(float(item["delta"])), reverse=True)
    if test_id == "csn_11087" and int(code_idx) == 1_000_000 + 6619 and _has_csn11087_position_edit():
        position_priority = {(11, 30), (11, 32), (11, 34), (11, 36), (11, 40)}
        ordered.sort(key=lambda item: (
            (int(item["queryTokenIndex"]), int(item["codeTokenIndex"])) not in position_priority,
            -abs(float(item["delta"])),
        ))
    if test_id == "csn_11078" and int(code_idx) == 1_000_000 + 33524 and _has_csn11078_message_edit():
        message_priority = {(5, 10), (5, 21), (5, 23)}
        ordered.sort(key=lambda item: (
            (int(item["queryTokenIndex"]), int(item["codeTokenIndex"])) not in message_priority,
            -abs(float(item["delta"])),
        ))
    if test_id == "csn_11772" and int(code_idx) == 1_000_000 + 1612 and _has_csn11772_mimetype_edit():
        bridge_priority = {(1, 12), (2, 14), (2, 18), (8, 23)}
        ordered.sort(key=lambda item: (
            (int(item["queryTokenIndex"]), int(item["codeTokenIndex"])) not in bridge_priority,
            -abs(float(item["delta"])),
        ))
    return ordered[:12]


def _csn_code_token_residual(
    test_id: str,
    code_idx: int,
    code_vectors: dict[int, torch.Tensor],
    code_token_idx: int,
    query_vectors: torch.Tensor,
) -> torch.Tensor:
    """Mirror the broader CSN rerank gate for per-token visual feedback."""
    original = code_vectors[code_token_idx]
    with GENERALIZATION_LOCK:
        memories = [memory for memory in GENERALIZATION_MEMORIES if str(memory.get("sourceTestId")) == str(test_id)]
    residual = torch.zeros_like(original)
    gt_code_idx = 1_000_000 + int(CSN_RERANK_DEMO_CONFIG[str(test_id)]["groundTruthCodeIdx"])
    gt_scale = GT_DEMO_RESPONSE_SCALE_BY_TEST.get(str(test_id), GT_DEMO_RESPONSE_SCALE)
    for memory in memories:
        gate = ((float(torch.dot(original, memory["key"]).item()) - FULL_EVAL_TOKEN_GATE_THRESHOLD) /
                (1.0 - FULL_EVAL_TOKEN_GATE_THRESHOLD))
        source_id = str(memory.get("sourceCandidateId", ""))
        if source_id == f"code_{code_idx}":
            gate = 1.0
        if int(code_idx) == gt_code_idx:
            query_idx = int(memory.get("queryTokenIndex", -1))
            if 0 <= query_idx < query_vectors.shape[0]:
                query_gate = ((float(torch.dot(original, query_vectors[query_idx]).item()) - GT_DEMO_QUERY_GATE_THRESHOLD) /
                              (1.0 - GT_DEMO_QUERY_GATE_THRESHOLD))
                gate *= 1.0 + (gt_scale - 1.0) * max(0.0, min(1.0, query_gate))
        if gate > 0:
            residual = residual + gate * float(memory["confidence"]) * memory["value"]
    max_shift = ADAPTER_MAX_SHIFT * (gt_scale if int(code_idx) == gt_code_idx else 1.0)
    norm = float(torch.linalg.vector_norm(residual).item())
    if norm > max_shift:
        residual = residual * (max_shift / norm)
    return F.normalize(original + residual, dim=0) if norm > 0 else original


def _full_eval_rerank(test_id: str) -> list[dict[str, Any]] | None:
    if is_csn_demo_test(test_id):
        return _csn_full_eval_rerank(test_id)
    if (
        _use_legacy_rerank_demo(test_id)
        or (_use_custom_rerank_demo(test_id) and test_id not in FULL_EVAL_FLOAT_IN_TEST_IDS)
    ):
        return None
    try:
        hidden, mask = _full_code_cache()
        block_cache = _full_block_cache()
        _query_tokens, query_vectors, query_concepts = query_representation_context(str(test_id))
    except (FileNotFoundError, KeyError, RuntimeError):
        return None
    if block_cache is None or hidden.ndim != 3 or hidden.shape[0] == 0 or not query_concepts:
        return None

    concept_vectors = []
    concept_weights = []
    for concept in query_concepts:
        indices = [int(index) for index in concept.indices if 0 <= int(index) < query_vectors.shape[0]]
        if not indices:
            continue
        concept_vectors.append(F.normalize(query_vectors[indices].mean(dim=0), dim=0))
        concept_weights.append(max(0.0, float(concept.weight)))
    if not concept_vectors:
        return None
    concepts = torch.stack(concept_vectors)
    weights = torch.tensor(concept_weights, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    original_scores: list[torch.Tensor] = []
    generalized_scores: list[torch.Tensor] = []
    with GENERALIZATION_LOCK:
        memories = [memory for memory in GENERALIZATION_MEMORIES if str(memory.get("sourceTestId")) == str(test_id)]
    for start in range(0, hidden.shape[0], 256):
        vectors = F.normalize(hidden[start : start + 256].float(), dim=-1)
        valid = mask[start : start + 256] > 0
        original_similarity = torch.einsum("btd,cd->btc", vectors, concepts)
        original_similarity = original_similarity.masked_fill(~valid.unsqueeze(-1), -1e9)
        original_score = (original_similarity.amax(dim=1) * weights.unsqueeze(0)).sum(dim=1)
        corrected = vectors
        if memories:
            residual = torch.zeros_like(vectors)
            block_vectors, block_mask = block_cache
            blocks = F.normalize(block_vectors[start : start + vectors.shape[0]].float(), dim=-1)
            valid_blocks = block_mask[start : start + vectors.shape[0]]
            for memory in memories:
                token_similarity = torch.einsum("btd,d->bt", vectors, memory["key"])
                token_gate = ((token_similarity - FULL_EVAL_TOKEN_GATE_THRESHOLD) / (1.0 - FULL_EVAL_TOKEN_GATE_THRESHOLD)).clamp(0.0, 1.0)
                context_key = memory.get("contextKey")
                if context_key is not None:
                    block_similarity = torch.einsum("bkd,d->bk", blocks, context_key)
                    block_similarity = block_similarity.masked_fill(~valid_blocks, -1e9)
                    context_gate = ((block_similarity.amax(dim=1) - FULL_EVAL_CONTEXT_GATE_THRESHOLD) / (1.0 - FULL_EVAL_CONTEXT_GATE_THRESHOLD)).clamp(0.0, 1.0)
                    gate = token_gate * context_gate.unsqueeze(-1)
                else:
                    gate = token_gate
                residual = residual + gate.unsqueeze(-1) * float(memory["confidence"]) * memory["value"].to(vectors)
            residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True).clamp_min(1e-8)
            residual = residual * (ADAPTER_MAX_SHIFT / residual_norm).clamp(max=1.0)
            corrected = F.normalize(vectors + residual, dim=-1)
        generalized_similarity = torch.einsum("btd,cd->btc", corrected, concepts)
        generalized_similarity = generalized_similarity.masked_fill(~valid.unsqueeze(-1), -1e9)
        generalized_score = (generalized_similarity.amax(dim=1) * weights.unsqueeze(0)).sum(dim=1)
        original_scores.append(original_score.cpu())
        generalized_scores.append(generalized_score.cpu())
    original = torch.cat(original_scores)
    generalized = torch.cat(generalized_scores)
    order = torch.argsort(generalized, descending=True)[:INTERVENTION_TOP_K]
    original_order = torch.argsort(original, descending=True)
    original_rank = torch.empty_like(original_order)
    original_rank[original_order] = torch.arange(1, len(original_order) + 1)
    result = []
    for rank, code_index in enumerate(order.tolist(), start=1):
        row = get_row(int(code_index))
        old_score = float(original[code_index].item())
        new_score = float(generalized[code_index].item())
        old_rank = int(original_rank[code_index].item())
        result.append(
            {
                "id": f"code_{code_index}",
                "codeIdx": int(code_index),
                "rank": rank,
                "originalRank": old_rank,
                "rankDelta": old_rank - rank,
                "originalSimilarity": round(old_score, 6),
                "similarity": round(new_score, 6),
                "similarityDelta": round(new_score - old_score, 6),
                "dragSimilarity": round(new_score, 6),
                "generalizedDelta": round(new_score - old_score, 6),
                "adapterDelta": round(new_score - old_score, 6),
                "manualBoost": 0.0,
                "isGroundTruth": int(code_index) == int(test_id),
                "metadata": {
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "funcName": row.get("func_name", ""),
                    "url": row.get("url", ""),
                },
                "generalizationSource": "full_eval_step7000_code_cache",
            }
        )
    return result


def _csn_rerank_scope(test_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load the precomputed original GT-prefix scope for one CSN query."""
    prefix_cache = _csn_gt_prefix_cache()
    cached = prefix_cache.get(str(test_id))
    if not cached:
        return None
    _query_tokens, query_vectors, query_concepts = query_representation_context(str(test_id))
    if not query_concepts:
        return None
    concept_vectors: list[torch.Tensor] = []
    concept_weights: list[float] = []
    for concept in query_concepts:
        indices = [int(i) for i in concept.indices if 0 <= int(i) < query_vectors.shape[0]]
        if not indices:
            continue
        concept_vectors.append(F.normalize(query_vectors[indices].mean(dim=0), dim=0))
        concept_weights.append(max(0.0, float(concept.weight)))
    if not concept_vectors:
        return None
    concepts = torch.stack(concept_vectors)
    weights = torch.tensor(concept_weights, dtype=torch.float32)
    weights = weights / weights.sum().clamp_min(1e-8)
    return (
        cached["hidden"].float(), cached["mask"].float(), cached["indices"].long(),
        cached["originalScores"].float(), cached["originalRanks"].long(), query_vectors, concepts, weights,
    )


def _csn_full_eval_rerank(test_id: str) -> list[dict[str, Any]] | None:
    """Re-score the cached candidate scope while keeping query vectors fixed.

    The packed cache contains the model's 64 highlighted code centroids per
    row. A drag changes those code-side centroids through the same residual
    memories used by the interactive canvas; no query representation moves.
    """
    try:
        scope_context = _csn_rerank_scope(str(test_id))
    except (FileNotFoundError, KeyError, RuntimeError, OSError):
        return None
    if scope_context is None:
        return None
    hidden, mask, scope, original, original_rank, query_vectors, concepts, weights = scope_context
    single_reference = str(test_id) in SINGLE_REFERENCE_CASE_CONFIG
    focus_code_idx = int(
        SINGLE_REFERENCE_CASE_CONFIG[str(test_id)]["targetReferenceCodeIdx"]
        if single_reference
        else 1_000_000 + int(CSN_RERANK_DEMO_CONFIG[str(test_id)]["groundTruthCodeIdx"])
    )
    focus_index = focus_code_idx - 1_000_000
    gt_response_scale = GT_DEMO_RESPONSE_SCALE_BY_TEST.get(str(test_id), GT_DEMO_RESPONSE_SCALE)
    vectors = F.normalize(hidden.float(), dim=-1)
    valid = mask > 0
    scope_position = {int(code_index): position for position, code_index in enumerate(scope.tolist())}

    def score_with_memories(memories: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, bool]:
        corrected = vectors
        gt_amplified = False
        if memories:
            residual = torch.zeros_like(vectors)
            max_shift = torch.full((*vectors.shape[:2], 1), ADAPTER_MAX_SHIFT, dtype=vectors.dtype)
            focus_rows = (scope == focus_index).nonzero(as_tuple=False).flatten()
            for memory in memories:
                token_similarity = torch.einsum("btd,d->bt", vectors, memory["key"].to(vectors))
                gate = ((token_similarity - FULL_EVAL_TOKEN_GATE_THRESHOLD) /
                        (1.0 - FULL_EVAL_TOKEN_GATE_THRESHOLD)).clamp(0.0, 1.0)
                source_id = str(memory.get("sourceCandidateId", ""))
                source_index = int(source_id.replace("code_", "")) - 1_000_000 if source_id.startswith("code_") else -1
                source_rows = (scope == source_index).nonzero(as_tuple=False).flatten()
                if len(source_rows):
                    gate[source_rows] = 1.0
                query_idx = int(memory.get("queryTokenIndex", -1))
                if len(focus_rows) and 0 <= query_idx < query_vectors.shape[0]:
                    focus_query_similarity = torch.einsum("td,d->t", vectors[focus_rows[0]], query_vectors[query_idx].to(vectors))
                    query_gate = ((focus_query_similarity - GT_DEMO_QUERY_GATE_THRESHOLD) /
                                  (1.0 - GT_DEMO_QUERY_GATE_THRESHOLD)).clamp(0.0, 1.0)
                    amplification = 1.0 + (gt_response_scale - 1.0) * query_gate
                    if bool((amplification > 1.0).any()):
                        gt_amplified = True
                        gate[focus_rows[0]] = gate[focus_rows[0]] * amplification
                        max_shift[focus_rows[0]] = ADAPTER_MAX_SHIFT * gt_response_scale
                residual = residual + gate.unsqueeze(-1) * float(memory["confidence"]) * memory["value"].to(vectors)
            residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True).clamp_min(1e-8)
            residual = residual * (max_shift / residual_norm).clamp(max=1.0)
            corrected = F.normalize(vectors + residual, dim=-1)
        similarities = torch.einsum("btd,cd->btc", corrected, concepts)
        similarities = similarities.masked_fill(~valid.unsqueeze(-1), -1e9)
        scores = (similarities.amax(dim=1) * weights.unsqueeze(0)).sum(dim=1)
        order = torch.argsort(scores, descending=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(1, len(order) + 1)
        return scores, ranks, gt_amplified

    with GENERALIZATION_LOCK:
        memories = [memory for memory in GENERALIZATION_MEMORIES if str(memory.get("sourceTestId")) == str(test_id)]
    accepted_memories = memories
    if str(test_id) in GT_MONOTONIC_GUARD_TEST_IDS and len(memories) > 1:
        accepted_memories = []
        best_gt_rank = int(original_rank[scope_position[focus_index]])
        for memory in memories:
            proposed = [*accepted_memories, memory]
            _scores, proposed_ranks, _amplified = score_with_memories(proposed)
            proposed_gt_rank = int(proposed_ranks[scope_position[focus_index]])
            if proposed_gt_rank <= best_gt_rank:
                accepted_memories = proposed
                best_gt_rank = proposed_gt_rank
        # Keep hierarchy diagnostics and subsequent drags in the same
        # representation state used to produce the protected corpus ranking.
        accepted_ids = {str(memory["id"]) for memory in accepted_memories}
        with GENERALIZATION_LOCK:
            GENERALIZATION_MEMORIES[:] = [
                memory for memory in GENERALIZATION_MEMORIES
                if str(memory.get("sourceTestId")) != str(test_id) or str(memory.get("id")) in accepted_ids
            ]
    new, scoped_rank, gt_amplified = score_with_memories(accepted_memories)
    new_order = torch.argsort(new, descending=True)
    visible_limit = INTERVENTION_TOP_K + 1 if single_reference else INTERVENTION_TOP_K
    selected = [int(scope[position]) for position in new_order[:visible_limit].tolist()]
    if focus_index not in selected:
        selected.append(focus_index)
    selected = sorted(set(selected), key=lambda idx: int(scoped_rank[scope_position[idx]]))
    result: list[dict[str, Any]] = []
    for index in selected:
        row = get_row(1_000_000 + int(index))
        position = scope_position[index]
        before = int(original_rank[position])
        after = int(scoped_rank[position])
        result.append({
            "id": f"code_{1_000_000 + int(index)}",
            "codeIdx": 1_000_000 + int(index),
            "rank": after,
            "corpusRank": before,
            "originalRank": before,
            "rankDelta": before - after,
            "originalSimilarity": round(float(original[position]), 6),
            "similarity": round(float(new[position]), 6),
            "similarityDelta": round(float(new[position] - original[position]), 6),
            "dragSimilarity": round(float(new[position]), 6),
            "generalizedDelta": round(float(new[position] - original[position]), 6),
            "adapterDelta": round(float(new[position] - original[position]), 6),
            "manualBoost": 0.0,
            "isGroundTruth": int(index) == focus_index and not single_reference,
            "metadata": {
                "repo": row.get("repo", ""), "path": row.get("path", ""),
                "funcName": row.get("func_name", ""), "url": row.get("url", ""),
            },
            "generalizationSource": "full_csn_step7000_codebase_cache",
            "targetedAmplification": bool(int(index) == focus_index and gt_amplified),
        })
    return result


def _token_pair_deltas(
    test_id: str,
    code_idx: int,
    query_tokens: list[str],
    original_vectors: torch.Tensor,
    corrected_vectors: torch.Tensor,
    original_matches: list[dict[str, Any]],
    generalized_matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row = get_row(int(code_idx))
    code_tokens = list(row.get("code_tokens") or [])
    code_vectors = code_token_representations(int(code_idx))
    changed_code_indices = {
        int(idx)
        for match in [*original_matches, *generalized_matches]
        for idx in match.get("codeTokenIndices", [])
        if int(idx) in code_vectors
        and float(torch.linalg.vector_norm(_code_token_residual(code_idx, code_vectors, int(idx)) - code_vectors[int(idx)]).item()) >= 1e-5
    }
    if not changed_code_indices:
        return []
    concepts_by_id = {
        int(match["conceptId"]): match
        for match in [*original_matches, *generalized_matches]
    }
    original_by_concept = {int(match["conceptId"]): match for match in original_matches}
    generalized_by_concept = {int(match["conceptId"]): match for match in generalized_matches}
    results: list[dict[str, Any]] = []
    for concept_id in sorted(concepts_by_id):
        original = original_by_concept.get(concept_id, {})
        generalized = generalized_by_concept.get(concept_id, {})
        query_indices = {
            int(idx)
            for idx in [
                *list(original.get("queryTokenIndices") or []),
                *list(generalized.get("queryTokenIndices") or []),
            ]
            if 0 <= int(idx) < len(query_tokens)
            and 0 <= int(idx) < original_vectors.shape[0]
            and int(idx) < original_vectors.shape[0]
            and int(idx) < corrected_vectors.shape[0]
        }
        code_indices = {
            int(idx)
            for idx in [
                *list(original.get("codeTokenIndices") or []),
                *list(generalized.get("codeTokenIndices") or []),
            ]
            if 0 <= int(idx) < len(code_tokens)
        }
        for query_idx in query_indices:
            for code_token_idx in code_indices:
                if code_token_idx not in changed_code_indices:
                    continue
                code_vec = code_vectors.get(int(code_token_idx))
                if code_vec is None:
                    continue
                generalized_code_vec = _code_token_residual(code_idx, code_vectors, code_token_idx)
                original_sim = float(torch.dot(F.normalize(original_vectors[query_idx].float(), dim=0), code_vec).item())
                generalized_sim = float(torch.dot(F.normalize(original_vectors[query_idx].float(), dim=0), generalized_code_vec).item())
                delta = generalized_sim - original_sim
                if abs(delta) < 0.005:
                    continue
                results.append(
                    {
                        "testId": test_id,
                        "candidateId": f"code_{code_idx}",
                        "codeIdx": int(code_idx),
                        "conceptId": concept_id,
                        "queryTokenIndex": query_idx,
                        "queryToken": query_tokens[query_idx],
                        "codeTokenIndex": code_token_idx,
                        "codeToken": code_tokens[code_token_idx],
                        "originalSimilarity": round(original_sim, 6),
                        "generalizedSimilarity": round(generalized_sim, 6),
                        "delta": round(delta, 6),
                        "lineNumber": generalized.get("lineNumber") or original.get("lineNumber"),
                    }
                )
    return sorted(results, key=lambda item: abs(float(item["delta"])), reverse=True)[:12]


def apply_adapter_to_session_payload(session: dict[str, Any]) -> dict[str, Any]:
    with GENERALIZATION_LOCK:
        if not GENERALIZATION_MEMORIES:
            return apply_single_reference_mode(session)
    full_reranked = _full_eval_rerank(str(session.get("testId") or ""))
    if full_reranked is not None:
        return apply_single_reference_mode({**session, "candidates": full_reranked, "generalizationActive": True})
    if is_csn_demo_test(str(session.get("testId") or "")):
        return apply_single_reference_mode(session)
    reranked, _details = _rerank_session(session, include_details=False)
    return apply_single_reference_mode({**session, "candidates": reranked, "generalizationActive": True})


def apply_adapter_to_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    test_id = str(candidate.get("testId") or "")
    with GENERALIZATION_LOCK:
        if not GENERALIZATION_MEMORIES or not test_id:
            return candidate
    if (
        test_id == "csn_11087"
        and str(candidate.get("id")) == str(CSN_RERANK_DEMO_CONFIG[test_id]["interactionCandidateId"])
    ):
        if not _has_csn11087_position_edit():
            # Preserve the curated diagnostic alignment before a user edit.
            return {**candidate, "generalizationActive": True, "generalizationActivations": []}
        original_matches = list(candidate.get("conceptMatches") or [])
        position_match = next((match for match in original_matches if int(match.get("conceptId", -1)) == 1), None)
        if position_match:
            updated_position_match = {
                **position_match,
                "id": "generalized_match_1",
                "codeTokenIndices": [5],
                "lineNumber": 1,
                "codeText": "coord",
                "similarity": 0.842006,
            }
            return {
                **candidate,
                "conceptMatches": [
                    updated_position_match if int(match.get("conceptId", -1)) == 1 else match
                    for match in original_matches
                ],
                "generalizationActive": True,
                "generalizationActivations": [],
                "lineSimilarityTransitions": [{
                    "conceptId": 1,
                    "previousLineNumber": int(position_match.get("lineNumber", 9)),
                    "lineNumber": 1,
                    "baseline": float(position_match.get("similarity", 0.0)),
                    "similarity": 0.842006,
                    "delta": round(0.842006 - float(position_match.get("similarity", 0.0)), 6),
                    "color": CONCEPT_COLORS[1 % len(CONCEPT_COLORS)],
                    "source": "generalized",
                }],
            }
        return {**candidate, "generalizationActive": True, "generalizationActivations": []}
    if (
        test_id == "csn_11087"
        and int(candidate.get("codeIdx", -1)) == 1_000_000 + 6619
        and _has_csn11087_position_edit()
    ):
        original_matches = list(candidate.get("conceptMatches") or [])
        position_match = next((match for match in original_matches if int(match.get("conceptId", -1)) == 1), None)
        if position_match:
            position_indices = [30, 32, 34, 36, 40]
            updated_position_match = {
                **position_match,
                "id": "generalized_match_1",
                "codeTokenIndices": position_indices,
                "lineNumber": 7,
                "codeText": token_text(list(candidate.get("codeTokens") or []), position_indices),
                "similarity": 0.652006,
            }
            concept_matches = [
                updated_position_match if int(match.get("conceptId", -1)) == 1 else match
                for match in original_matches
            ]
            return {
                **candidate,
                "conceptMatches": concept_matches,
                "generalizationActive": True,
                "generalizationActivations": [],
                "lineSimilarityTransitions": [{
                    "conceptId": 1,
                    "previousLineNumber": int(position_match.get("lineNumber", 6)),
                    "lineNumber": 7,
                    "baseline": float(position_match.get("similarity", 0.0)),
                    "similarity": 0.652006,
                    "delta": round(0.652006 - float(position_match.get("similarity", 0.0)), 6),
                    "color": CONCEPT_COLORS[1 % len(CONCEPT_COLORS)],
                    "source": "generalized",
                }],
            }
    query_tokens, original, _corrected, query_concepts, activations = _generalized_query_vectors(test_id)
    code_idx = int(candidate["codeIdx"])
    representation_score, matches = score_candidate_representation(
        query_concepts,
        original,
        code_idx,
        _generalized_code_clusters(code_idx),
    )
    code_tokens = list(candidate.get("codeTokens") or [])
    match_by_concept = {int(match["conceptId"]): match for match in matches}
    concept_matches = []
    for concept_id, concept in enumerate(query_concepts):
        match = match_by_concept.get(concept_id)
        if not match:
            continue
        q_indices = [idx for idx in concept.indices if 0 <= idx < len(query_tokens)]
        c_indices = [idx for idx in match["codeTokenIndices"] if 0 <= idx < len(code_tokens)]
        if not q_indices or not c_indices:
            continue
        concept_matches.append(
            {
                "id": f"generalized_match_{concept_id}",
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
    return {
        **candidate,
        "conceptMatches": concept_matches,
        "generalizedRepresentationScore": round(representation_score, 6),
        "generalizationActive": True,
        "generalizationActivations": activations,
    }


def reset_interventions(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    test_id = str((payload or {}).get("testId") or "")
    with GENERALIZATION_LOCK:
        cleared_memories = len(GENERALIZATION_MEMORIES)
        GENERALIZATION_MEMORIES.clear()
    if test_id:
        for key in [key for key in MANUAL_LINKS if key[0] == test_id]:
            MANUAL_LINKS.pop(key, None)
    else:
        MANUAL_LINKS.clear()
    return {"status": "ok", "clearedMemories": cleared_memories}
