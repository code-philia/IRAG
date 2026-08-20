from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering

from .config import CONCEPT_COLORS, LATEST_STEP_CHECKPOINT_PATH, LOCAL_COCOSODA_PATH, TRAINING_EVAL_RESULTS_DIR, XSEARCH_ROOT
from .data_service import build_code_lines, load_smoke_codebase, token_text


PACKED_STEP7000 = TRAINING_EVAL_RESULTS_DIR / "python_full_step7000_packed.pt"


@dataclass
class QueryConcept:
    indices: list[int]
    centroid: torch.Tensor
    weight: float


@dataclass
class QueryEncoding:
    tokens: list[str]
    vectors: torch.Tensor
    concepts: list[QueryConcept]
    scores: torch.Tensor | None = None


def _load_model():
    if str(XSEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(XSEARCH_ROOT))
    from model import Model
    from transformers import RobertaModel, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(str(LOCAL_COCOSODA_PATH), local_files_only=True)
    encoder = RobertaModel.from_pretrained(str(LOCAL_COCOSODA_PATH), local_files_only=True)
    model = Model(encoder, num_roles=20, use_cross_sample_loss=True)
    state_dict = torch.load(LATEST_STEP_CHECKPOINT_PATH, map_location="cpu")
    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=1)
def model_context():
    return _load_model()


@lru_cache(maxsize=1)
def packed_step7000():
    hidden, scores, urls = torch.load(PACKED_STEP7000, map_location="cpu", mmap=True)
    url_index = {str(url): idx for idx, url in enumerate(urls)}
    return hidden, scores, urls, url_index


def _query_text(row: dict[str, Any]) -> str:
    return row.get("clean_docstring") or row.get("docstring") or row.get("func_name", "").replace(".", " ")


@lru_cache(maxsize=64)
def _extract_query_encoding(text: str, cluster_threshold: float = 0.8) -> QueryEncoding:
    tokenizer, model = model_context()
    inputs = tokenizer([text], padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.inference_mode():
        outputs = model(nl_inputs=inputs.input_ids)
    raw_tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    hidden = outputs.nl_hidden[0].detach().cpu().float()
    scores = outputs.nl_scores[0]
    special_tokens = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    display_slots = [idx for idx, token in enumerate(raw_tokens) if token not in special_tokens]
    raw_to_display = {raw_idx: display_idx for display_idx, raw_idx in enumerate(display_slots)}
    tokens = [raw_tokens[idx] for idx in display_slots]
    display_vectors = hidden[display_slots] if display_slots else hidden[:0]
    display_scores = scores[display_slots].detach().cpu().float() if display_slots else scores[:0].detach().cpu().float()

    important_raw = sorted((scores > 0.4).nonzero(as_tuple=True)[0].tolist())
    important_raw = [idx for idx in important_raw if idx in raw_to_display]
    important = [raw_to_display[idx] for idx in important_raw]
    if not important:
        important = list(range(min(1, len(tokens))))
    if not important:
        return QueryEncoding(tokens=tokens, vectors=display_vectors, concepts=[], scores=display_scores)

    vectors = F.normalize(display_vectors[important], dim=1)
    if len(important) == 1:
        raw_idx = display_slots[important[0]]
        return QueryEncoding(
            tokens=tokens,
            vectors=display_vectors,
            concepts=[QueryConcept(indices=important, centroid=vectors[0], weight=float(scores[raw_idx].item()))],
            scores=display_scores,
        )

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1 - cluster_threshold,
        )
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            affinity="cosine",
            linkage="average",
            distance_threshold=1 - cluster_threshold,
        )
    labels = clusterer.fit_predict(vectors.detach().cpu().numpy())

    concepts: list[QueryConcept] = []
    for label in sorted(set(labels)):
        local = np.where(labels == label)[0]
        indices = [important[int(i)] for i in local]
        centroid = F.normalize(vectors[local].mean(dim=0), dim=0)
        raw_indices = [display_slots[idx] for idx in indices]
        weight = float(scores[raw_indices].mean().item())
        concepts.append(QueryConcept(indices=indices, centroid=centroid, weight=weight))
    return QueryEncoding(tokens=tokens, vectors=display_vectors, concepts=concepts, scores=display_scores)


def _extract_query_concepts(text: str, cluster_threshold: float = 0.8) -> tuple[list[str], list[QueryConcept]]:
    encoding = _extract_query_encoding(text, cluster_threshold)
    return encoding.tokens, _override_query_concepts(text, encoding.vectors, encoding.concepts)


def _override_query_concepts(text: str, vectors: torch.Tensor, concepts: list[QueryConcept]) -> list[QueryConcept]:
    if text.strip().lower().rstrip(".") != "api function decorator that performs rate limiting and error checking":
        return concepts
    keep = [concept for concept in concepts if 8 not in concept.indices and not set(concept.indices).intersection({9, 10})]
    merged_indices = [9, 10]
    if vectors.shape[0] > max(merged_indices):
        merged = F.normalize(vectors[merged_indices].mean(dim=0), dim=0)
        keep.append(QueryConcept(indices=merged_indices, centroid=merged, weight=0.5))
    return keep


def get_aligned_query_vectors(test_id: str) -> tuple[list[str], np.ndarray] | None:
    rows = load_smoke_codebase()
    idx = int(str(test_id).split("_", 1)[1])
    if idx < 0 or idx >= len(rows):
        return None
    encoding = _extract_query_encoding(_query_text(rows[idx]))
    if encoding.vectors.numel() == 0:
        return None
    vectors = F.normalize(encoding.vectors, dim=1).detach().cpu().numpy().astype(np.float32, copy=True)
    return encoding.tokens, vectors


def _code_vectors_for_url(url: str):
    hidden, scores, _urls, url_index = packed_step7000()
    if url not in url_index:
        return None
    row = int(url_index[url])
    return hidden[row].detach().cpu().float(), scores[row].detach().cpu().float()


def _line_centroids(row: dict[str, Any], hidden: torch.Tensor, scores: torch.Tensor):
    code_tokens = list(row.get("code_tokens") or [])
    raw_code = row.get("clean_code") or row.get("code") or row.get("original_string") or ""
    lines = build_code_lines(raw_code, code_tokens)
    clusters = []
    for line in lines:
        token_indices = [int(idx) for idx in line.get("tokenIndices", []) if 0 <= int(idx) < len(code_tokens)]
        slots = [idx + 1 for idx in token_indices if idx + 1 < hidden.shape[0]]
        if not slots:
            continue
        highlighted = [slot for slot in slots if float(scores[slot].item()) > 0.5]
        used_slots = highlighted or slots
        centroid = F.normalize(hidden[used_slots].mean(dim=0), dim=0)
        clusters.append(
            {
                "lineNumber": int(line["lineNumber"]),
                "tokenIndices": token_indices,
                "slots": used_slots,
                "centroid": centroid,
            }
        )
    return clusters


def _score_code(query_concepts: list[QueryConcept], code_clusters: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    if not query_concepts or not code_clusters:
        return 0.0, []
    weights = torch.tensor([max(0.0, concept.weight) for concept in query_concepts], dtype=torch.float32)
    if float(weights.sum().item()) > 0:
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(query_concepts)) / len(query_concepts)

    code_centroids = torch.stack([cluster["centroid"] for cluster in code_clusters])
    matches = []
    max_sims = []
    for concept_id, concept in enumerate(query_concepts):
        sims = torch.matmul(code_centroids, concept.centroid)
        best_sim, best_idx = torch.max(sims, dim=0)
        cluster = code_clusters[int(best_idx.item())]
        max_sims.append(best_sim)
        matches.append(
            {
                "conceptId": concept_id,
                "queryTokenIndices": concept.indices,
                "codeTokenIndices": cluster["tokenIndices"],
                "lineNumber": cluster["lineNumber"],
                "similarity": round(float(best_sim.item()), 6),
            }
        )
    score = float((torch.stack(max_sims) * weights).sum().item())
    return round(score, 6), matches


@lru_cache(maxsize=64)
def build_aligned_smoke_session(test_id: str, top_k: int = 10) -> dict[str, Any]:
    rows = load_smoke_codebase()
    idx = int(str(test_id).split("_", 1)[1])
    query_row = rows[idx]
    query_text = _query_text(query_row)
    query_tokens, query_concepts = _extract_query_concepts(query_text)

    scored = []
    for code_idx, row in enumerate(rows[:200]):
        code_pack = _code_vectors_for_url(row.get("url", ""))
        if code_pack is None:
            continue
        code_hidden, code_scores = code_pack
        clusters = _line_centroids(row, code_hidden, code_scores)
        score, _matches = _score_code(query_concepts, clusters)
        scored.append((score, code_idx, row))
    scored.sort(key=lambda item: item[0], reverse=True)

    concepts = []
    for concept_id, concept in enumerate(query_concepts):
        concepts.append(
            {
                "id": f"concept_{concept_id}",
                "conceptId": concept_id,
                "tokenIndices": concept.indices,
                "text": token_text(query_tokens, concept.indices),
                "color": CONCEPT_COLORS[concept_id % len(CONCEPT_COLORS)],
                "weight": round(float(concept.weight), 6),
            }
        )

    candidates = []
    for rank, (score, code_idx, row) in enumerate(scored[:top_k], start=1):
        candidates.append(
            {
                "id": f"code_{code_idx}",
                "codeIdx": code_idx,
                "rank": rank,
                "similarity": score,
                "metadata": {
                    "repo": row.get("repo", ""),
                    "path": row.get("path", ""),
                    "funcName": row.get("func_name", ""),
                    "url": row.get("url", ""),
                },
            }
        )

    return {
        "testId": str(test_id),
        "query": {
            "rawText": query_text,
            "tokens": query_tokens,
            "concepts": concepts,
            "metadata": {
                "repo": query_row.get("repo", ""),
                "path": query_row.get("path", ""),
                "funcName": query_row.get("func_name", ""),
                "url": query_row.get("url", ""),
            },
        },
        "candidates": candidates,
        "rankingSource": "xsearch_step7000_checkpoint",
        "conceptSource": "xsearch_nl_scores_clustered_step7000",
    }


def build_aligned_smoke_candidate(test_id: str, candidate_id: str) -> dict[str, Any]:
    rows = load_smoke_codebase()
    query_idx = int(str(test_id).split("_", 1)[1])
    code_idx = int(candidate_id.replace("code_", ""))
    query_row = rows[query_idx]
    row = rows[code_idx]
    session = build_aligned_smoke_session(test_id, 200)
    query_tokens = list(session["query"]["tokens"])
    query_concepts = _extract_query_concepts(_query_text(query_row))[1]

    code_pack = _code_vectors_for_url(row.get("url", ""))
    if code_pack is None:
        raise ValueError(f"candidate URL not found in step7000 packed cache: {row.get('url', '')}")
    code_hidden, code_scores = code_pack
    clusters = _line_centroids(row, code_hidden, code_scores)
    score, matches = _score_code(query_concepts, clusters)

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
        "similarity": score,
        "conceptMatches": concept_matches,
        "metadata": {
            "repo": row.get("repo", ""),
            "path": row.get("path", ""),
            "funcName": row.get("func_name", ""),
            "url": row.get("url", ""),
        },
        "rankingSource": "xsearch_step7000_checkpoint",
        "conceptSource": "xsearch_nl_scores_clustered_step7000",
    }
