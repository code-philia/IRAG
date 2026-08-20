from __future__ import annotations

import json
import os
import sys
import hashlib
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .aligned_xsearch_service import _extract_query_encoding, _query_text, model_context
from .config import GRADIENT_BATCH_CACHE_DIR, LATEST_STEP_CHECKPOINT_PATH, TRAIN_DATA_FILE, XSEARCH_ROOT
from .data_service import get_row
from .training_attribution_service import _score_training_evidence
from .user_study_aligned_service import _feature_to_batch, _roles


TRAIN_ROLES_PATH = XSEARCH_ROOT / "preprocess_dataset/role_tensor_python_train_256.npy"
GRADIENT_PARAM_NAMES = ("encoder.encoder.layer.11", "nl_highlight_layer", "code_highlight_layer", "role_embedding")
TRAIN_BATCH_SIZE = 32
LOSS_SCOPES = {"highlight_only", "highlight_plus_cross_sample_batch"}
CROSS_SAMPLE_EVIDENCE_LIMIT = 256


@lru_cache(maxsize=1)
def _train_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(TRAIN_DATA_FILE, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def _train_roles() -> torch.Tensor:
    arr = np.load(TRAIN_ROLES_PATH, mmap_mode="r")
    return torch.from_numpy(np.asarray(arr[:, :256]).copy()).long()


def _ensure_xsearch_path() -> None:
    if str(XSEARCH_ROOT) not in sys.path:
        sys.path.insert(0, str(XSEARCH_ROOT))


def _feature_for_train_index(train_index: int) -> Any:
    _ensure_xsearch_path()
    from dataloader import TextDataset

    tokenizer, _model = model_context()
    args = SimpleNamespace(nl_length=128, code_length=256, data_flow_length=0, lang="python", device="cpu")
    rows = _train_rows()
    if not (0 <= train_index < len(rows)):
        raise ValueError(f"Train index out of range: {train_index}")
    old_cwd = Path.cwd()
    os.chdir(XSEARCH_ROOT)
    try:
        return TextDataset.convert_examples_to_features((rows[train_index], tokenizer, args), compute_alignment=True)
    finally:
        os.chdir(old_cwd)


def _total_code_tokens(code_inputs: torch.Tensor, ori2cur_pos: torch.Tensor) -> int:
    code_tokens_2 = (code_inputs[0] == 2).nonzero().flatten()
    if len(code_tokens_2) == 0:
        return 255
    return int(code_tokens_2[0].item())


def _total_comment_tokens(nl_inputs: torch.Tensor) -> int:
    nl_tokens_2 = (nl_inputs[0] == 2).nonzero().flatten()
    if len(nl_tokens_2) == 0:
        return 127
    return int(nl_tokens_2[0].item())


def _feature_to_training_batch(feature: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    code_inputs, attn_mask, position_idx = _feature_to_batch(feature)
    nl_inputs = torch.tensor(feature.nl_ids).unsqueeze(0)
    ori2cur_pos_list = [[start, end] for start, end in feature.ori2cur_pos.values()]
    ori2cur_pos_list = ori2cur_pos_list + [[0, 0]] * (256 - len(ori2cur_pos_list))
    ori2cur_pos = torch.tensor(ori2cur_pos_list[:256]).unsqueeze(0)
    return code_inputs, attn_mask, position_idx, nl_inputs, ori2cur_pos


def _collate_training_features(features: list[Any]) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[Any],
    list[Any],
    list[Any],
]:
    code_inputs_list = []
    attn_mask_list = []
    position_idx_list = []
    nl_inputs_list = []
    ori2cur_pos_list = []
    match_list = []
    valid_code_spans_batch = []
    valid_comment_spans_batch = []
    for feature in features:
        code_inputs, attn_mask, position_idx, nl_inputs, ori2cur_pos = _feature_to_training_batch(feature)
        code_inputs_list.append(code_inputs[0])
        attn_mask_list.append(attn_mask[0])
        position_idx_list.append(position_idx[0])
        nl_inputs_list.append(nl_inputs[0])
        ori2cur_pos_list.append(ori2cur_pos[0])
        match_list.append(feature.concept_alignment)
        valid_code_spans_batch.append(feature.valid_code_spans or [])
        valid_comment_spans_batch.append(feature.valid_comment_spans or [])
    return (
        torch.stack(code_inputs_list),
        torch.stack(attn_mask_list),
        torch.stack(position_idx_list),
        torch.stack(nl_inputs_list),
        torch.stack(ori2cur_pos_list),
        match_list,
        valid_code_spans_batch,
        valid_comment_spans_batch,
    )


def _selected_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [param for name, param in model.named_parameters() if any(prefix in name for prefix in GRADIENT_PARAM_NAMES)]


def _flatten_grads(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    pieces = []
    for grad, param in zip(grads, params):
        if grad is None:
            pieces.append(torch.zeros(param.numel(), dtype=torch.float32))
        else:
            pieces.append(grad.detach().cpu().float().reshape(-1))
    if not pieces:
        return torch.empty(0)
    return torch.cat(pieces)


def _checkpoint_fingerprint() -> str:
    try:
        stat = LATEST_STEP_CHECKPOINT_PATH.stat()
        raw = f"{LATEST_STEP_CHECKPOINT_PATH}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        raw = str(LATEST_STEP_CHECKPOINT_PATH)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _batch_cache_path(batch_index: int, cross_sample_weight: float) -> Path:
    key_payload = {
        "checkpoint": _checkpoint_fingerprint(),
        "batchIndex": batch_index,
        "batchSize": TRAIN_BATCH_SIZE,
        "lossScope": "highlight_plus_cross_sample_batch",
        "crossSampleWeight": round(float(cross_sample_weight), 6),
        "parameterScope": list(GRADIENT_PARAM_NAMES),
    }
    key = hashlib.sha1(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    return GRADIENT_BATCH_CACHE_DIR / f"batch_{batch_index}_{key}.pt"


def _load_cached_batch_gradient(batch_index: int, cross_sample_weight: float) -> dict[str, Any] | None:
    path = _batch_cache_path(batch_index, cross_sample_weight)
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError, EOFError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("grad"), torch.Tensor):
        return None
    payload = dict(payload)
    payload["cacheHit"] = True
    payload["cachePath"] = str(path)
    return payload


def _save_cached_batch_gradient(batch: dict[str, Any], batch_index: int, cross_sample_weight: float) -> None:
    path = _batch_cache_path(batch_index, cross_sample_weight)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    payload = dict(batch)
    payload["grad"] = payload["grad"].detach().cpu().float()
    payload["cacheHit"] = False
    payload["cachePath"] = str(path)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _batch_gradient_cached(batch_index: int, cross_sample_weight: float) -> dict[str, Any] | None:
    cached = _load_cached_batch_gradient(batch_index, cross_sample_weight)
    if cached is not None:
        return cached
    batch = _batch_gradient(batch_index, cross_sample_weight)
    if batch is None:
        return None
    _save_cached_batch_gradient(batch, batch_index, cross_sample_weight)
    batch = dict(batch)
    batch["cacheHit"] = False
    batch["cachePath"] = str(_batch_cache_path(batch_index, cross_sample_weight))
    return batch


def _select_candidate_batches(evidence: dict[str, Any], max_batches: int) -> list[dict[str, Any]]:
    batches: dict[int, dict[str, Any]] = {}
    evidence_items: list[tuple[str, dict[str, Any]]] = []
    evidence_items.extend(("support", item) for item in evidence.get("supportingSamples", []))
    evidence_items.extend(("conflict", item) for item in evidence.get("conflictingSamples", []))
    for source, item in evidence_items:
        try:
            train_index = int(item.get("trainIndex"))
        except (TypeError, ValueError):
            continue
        batch_index = train_index // TRAIN_BATCH_SIZE
        entry = batches.setdefault(
            batch_index,
            {
                "batchIndex": batch_index,
                "hitCount": 0,
                "supportCount": 0,
                "conflictCount": 0,
                "exactTokenPairHits": 0,
                "scores": [],
                "sampleHits": [],
            },
        )
        score = float(item.get("score") or 0.0)
        query_similarity = float(item.get("querySimilarity") or 0.0)
        code_similarity = float(item.get("codeSimilarity") or 0.0)
        entry["hitCount"] += 1
        entry["supportCount"] += 1 if source == "support" else 0
        entry["conflictCount"] += 1 if source == "conflict" else 0
        entry["exactTokenPairHits"] += 1 if query_similarity >= 0.99 and code_similarity >= 0.99 else 0
        entry["scores"].append(score)
        entry["sampleHits"].append(
            {
                "trainIndex": train_index,
                "source": source,
                "score": round(score, 6),
                "querySimilarity": round(query_similarity, 6),
                "codeSimilarity": round(code_similarity, 6),
                "funcName": item.get("funcName", ""),
                "path": item.get("path", ""),
                "docstring": item.get("docstring", ""),
                "conceptText": item.get("conceptText", ""),
                "stepDesc": item.get("stepDesc", ""),
                "stepCode": item.get("stepCode", ""),
            }
        )

    ranked = []
    for entry in batches.values():
        scores = sorted((float(score) for score in entry.pop("scores")), reverse=True)
        max_score = scores[0] if scores else 0.0
        top3_sum = sum(scores[:3])
        hit_count = int(entry["hitCount"])
        exact_hits = int(entry["exactTokenPairHits"])
        entry["selectionScore"] = round(max_score + 0.5 * top3_sum + 0.3 * min(hit_count, 5) + 0.2 * min(exact_hits, 3), 6)
        entry["maxEvidenceScore"] = round(max_score, 6)
        entry["top3EvidenceScoreSum"] = round(top3_sum, 6)
        entry["batchStart"] = int(entry["batchIndex"]) * TRAIN_BATCH_SIZE
        entry["batchEnd"] = min(len(_train_rows()) - 1, int(entry["batchIndex"]) * TRAIN_BATCH_SIZE + TRAIN_BATCH_SIZE - 1)
        entry["sampleHits"] = sorted(entry["sampleHits"], key=lambda item: float(item["score"]), reverse=True)[:5]
        ranked.append(entry)
    return sorted(
        ranked,
        key=lambda item: (float(item["selectionScore"]), int(item["hitCount"]), float(item["maxEvidenceScore"])),
        reverse=True,
    )[:max_batches]


def _pair_gradient(test_id: str, code_idx: int, query_token_index: int, code_token_index: int, mode: str) -> tuple[torch.Tensor, float, float]:
    query_row = get_row(int(test_id))
    query_encoding = _extract_query_encoding(_query_text(query_row))
    if not (0 <= query_token_index < len(query_encoding.tokens)):
        raise ValueError("Query token index is out of range.")

    _ensure_xsearch_path()
    tokenizer, model = model_context()
    params = _selected_parameters(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in params:
        param.requires_grad_(True)

    from dataloader import TextDataset

    args = SimpleNamespace(nl_length=128, code_length=256, data_flow_length=0, lang="python", device="cpu")
    row = get_row(code_idx)
    old_cwd = Path.cwd()
    os.chdir(XSEARCH_ROOT)
    try:
        feature = TextDataset.convert_examples_to_features((row, tokenizer, args), compute_alignment=False)
    finally:
        os.chdir(old_cwd)
    code_inputs, attn_mask, position_idx = _feature_to_batch(feature)
    nl_inputs = tokenizer([_query_text(query_row)], padding=True, truncation=True, max_length=128, return_tensors="pt").input_ids
    train_roles = _roles()[code_idx : code_idx + 1]
    code_outputs, nl_outputs = model(
        code_inputs=code_inputs,
        attn_mask=attn_mask,
        position_idx=position_idx,
        nl_inputs=nl_inputs,
        role_indices=train_roles,
        force_return_tuple=True,
    )

    raw_tokens = tokenizer.convert_ids_to_tokens(nl_inputs[0])
    special_tokens = {tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token}
    display_slots = [idx for idx, token in enumerate(raw_tokens) if token not in special_tokens]
    if query_token_index >= len(display_slots):
        raise ValueError("Query token index does not map to model slot.")
    q_slot = display_slots[query_token_index]

    span = feature.ori2cur_pos.get(int(code_token_index))
    if not span:
        raise ValueError("Code token index does not map to model subtokens.")
    c_slots = [slot for slot in range(int(span[0]) + 1, int(span[1]) + 1) if 0 <= slot < code_outputs.last_hidden_state.shape[1]]
    if not c_slots:
        raise ValueError("Code token has no valid subtoken slot.")

    q_hidden = F.normalize(nl_outputs.last_hidden_state[0, q_slot], dim=0)
    c_hidden = F.normalize(code_outputs.last_hidden_state[0, c_slots].mean(dim=0), dim=0)
    cosine = torch.dot(q_hidden, c_hidden)
    loss = -cosine if mode == "pull" else cosine
    grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    grad_vec = _flatten_grads(grads, params)
    return grad_vec, float(loss.detach().item()), float(cosine.detach().item())


def _train_sample_gradient(train_index: int) -> dict[str, Any] | None:
    tokenizer, model = model_context()
    params = _selected_parameters(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in params:
        param.requires_grad_(True)

    feature = _feature_for_train_index(train_index)
    if not feature.concept_alignment:
        return None
    code_inputs, attn_mask, position_idx, nl_inputs, ori2cur_pos = _feature_to_training_batch(feature)
    roles = _train_roles()[train_index : train_index + 1]
    code_outputs, nl_outputs = model(
        code_inputs=code_inputs,
        attn_mask=attn_mask,
        position_idx=position_idx,
        nl_inputs=nl_inputs,
        role_indices=roles,
        force_return_tuple=True,
    )
    total_code_tokens = _total_code_tokens(code_inputs, ori2cur_pos)
    total_comment_tokens = _total_comment_tokens(nl_inputs)
    total_loss, nl_loss, code_loss = model.compute_loss(
        code_inputs,
        code_outputs,
        nl_outputs,
        0,
        feature.concept_alignment,
        total_code_tokens,
        total_comment_tokens,
        roles,
        feature.valid_code_spans,
    )
    grads = torch.autograd.grad(total_loss, params, allow_unused=True, retain_graph=False)
    grad_vec = _flatten_grads(grads, params)
    rows = _train_rows()
    row = rows[train_index]
    return {
        "trainIndex": train_index,
        "batchIndex": train_index // TRAIN_BATCH_SIZE,
        "batchStart": (train_index // TRAIN_BATCH_SIZE) * TRAIN_BATCH_SIZE,
        "batchEnd": min(len(rows) - 1, (train_index // TRAIN_BATCH_SIZE) * TRAIN_BATCH_SIZE + TRAIN_BATCH_SIZE - 1),
        "grad": grad_vec,
        "loss": float(total_loss.detach().item()),
        "nlHighlightLoss": float(nl_loss.detach().item()),
        "codeHighlightLoss": float(code_loss.detach().item()),
        "url": row.get("url", ""),
        "path": row.get("path", ""),
        "funcName": row.get("func_name", ""),
        "docstring": str(row.get("clean_docstring") or row.get("docstring") or "")[:800],
        "code": str(row.get("original_string") or row.get("code") or "")[:1200],
        "validCommentSpans": feature.valid_comment_spans or [],
        "validCodeSpans": feature.valid_code_spans or [],
    }


def _batch_gradient(batch_index: int, cross_sample_weight: float) -> dict[str, Any] | None:
    _tokenizer, model = model_context()
    params = _selected_parameters(model)
    for param in model.parameters():
        param.requires_grad_(False)
    for param in params:
        param.requires_grad_(True)

    rows = _train_rows()
    batch_start = batch_index * TRAIN_BATCH_SIZE
    batch_end = min(len(rows), batch_start + TRAIN_BATCH_SIZE)
    if batch_start >= batch_end:
        return None
    train_indices = list(range(batch_start, batch_end))
    features = [_feature_for_train_index(idx) for idx in train_indices]
    (
        code_inputs,
        attn_mask,
        position_idx,
        nl_inputs,
        ori2cur_pos,
        match_list,
        valid_code_spans_batch,
        valid_comment_spans_batch,
    ) = _collate_training_features(features)
    roles = _train_roles()[batch_start:batch_end]
    code_outputs, nl_outputs = model(
        code_inputs=code_inputs,
        attn_mask=attn_mask,
        position_idx=position_idx,
        nl_inputs=nl_inputs,
        role_indices=roles,
        force_return_tuple=True,
    )

    total_attention_loss = torch.tensor(0.0)
    total_nl_highlight_loss = torch.tensor(0.0)
    total_code_highlight_loss = torch.tensor(0.0)
    total_code_tokens_list = []
    total_comment_tokens_list = []
    for batch_pos in range(code_inputs.size(0)):
        total_code_tokens = _total_code_tokens(code_inputs[batch_pos : batch_pos + 1], ori2cur_pos[batch_pos : batch_pos + 1])
        total_comment_tokens = _total_comment_tokens(nl_inputs[batch_pos : batch_pos + 1])
        total_loss, nl_loss, code_loss = model.compute_loss(
            code_inputs,
            code_outputs,
            nl_outputs,
            batch_pos,
            match_list[batch_pos],
            total_code_tokens,
            total_comment_tokens,
            roles,
            valid_code_spans_batch[batch_pos],
        )
        total_attention_loss = total_attention_loss + total_loss
        total_nl_highlight_loss = total_nl_highlight_loss + nl_loss
        total_code_highlight_loss = total_code_highlight_loss + code_loss
        total_code_tokens_list.append(total_code_tokens)
        total_comment_tokens_list.append(total_comment_tokens)

    attention_loss = total_attention_loss / max(1, code_inputs.size(0))
    nl_highlight_loss = total_nl_highlight_loss / max(1, code_inputs.size(0))
    code_highlight_loss = total_code_highlight_loss / max(1, code_inputs.size(0))
    cross_sample_loss = torch.tensor(0.0)
    if len(valid_code_spans_batch) > 1:
        cross_sample_loss = model.compute_cross_sample_contrastive_loss_with_filtering(
            nl_outputs.last_hidden_state,
            code_outputs.last_hidden_state,
            match_list,
            total_code_tokens_list,
            total_comment_tokens_list,
            valid_comment_spans_batch,
            valid_code_spans_batch,
            similarity_threshold=0.3,
            max_negative_samples_per_concept=50,
            code_inputs=code_inputs,
            nl_inputs=nl_inputs,
        )
    total_loss = attention_loss + cross_sample_loss * cross_sample_weight
    grads = torch.autograd.grad(total_loss, params, allow_unused=True, retain_graph=False)
    grad_vec = _flatten_grads(grads, params)
    batch_rows = rows[batch_start:batch_end]
    return {
        "batchIndex": batch_index,
        "batchStart": batch_start,
        "batchEnd": batch_end - 1,
        "trainIndices": train_indices,
        "grad": grad_vec,
        "loss": float(total_loss.detach().item()),
        "highlightLoss": float(attention_loss.detach().item()),
        "nlHighlightLoss": float(nl_highlight_loss.detach().item()),
        "codeHighlightLoss": float(code_highlight_loss.detach().item()),
        "crossSampleLoss": float(cross_sample_loss.detach().item()),
        "crossSampleWeight": cross_sample_weight,
        "sampleSummaries": [
            {
                "trainIndex": batch_start + offset,
                "funcName": row.get("func_name", ""),
                "path": row.get("path", ""),
                "url": row.get("url", ""),
                "docstring": str(row.get("clean_docstring") or row.get("docstring") or "")[:240],
            }
            for offset, row in enumerate(batch_rows)
        ],
    }


def build_gradient_attribution(payload: dict[str, Any]) -> dict[str, Any]:
    test_id = str(payload.get("testId") or "")
    candidate_id = str(payload.get("candidateId") or "")
    query_token_index = int(payload.get("queryTokenIndex"))
    code_token_index = int(payload.get("codeTokenIndex"))
    mode = str(payload.get("mode") or "pull")
    if mode not in {"pull", "push"}:
        raise ValueError("mode must be 'pull' or 'push'.")
    loss_scope = str(payload.get("lossScope") or "highlight_only")
    if loss_scope not in LOSS_SCOPES:
        raise ValueError("lossScope must be 'highlight_only' or 'highlight_plus_cross_sample_batch'.")
    top_k = max(1, min(10, int(payload.get("topK") or 5)))
    max_train_samples = max(1, min(64, int(payload.get("maxTrainSamples") or 24)))
    max_batches = max(1, min(16, int(payload.get("maxBatches") or max_train_samples)))
    cross_sample_weight = float(payload.get("crossSampleWeight") or 2.0)
    code_idx = int(candidate_id.replace("code_", ""))

    query_row = get_row(int(test_id))
    query_encoding = _extract_query_encoding(_query_text(query_row))
    code_tokens = list(get_row(code_idx).get("code_tokens") or [])
    query_token = query_encoding.tokens[query_token_index]
    code_token = code_tokens[code_token_index]

    evidence_limit = max_train_samples
    if loss_scope == "highlight_plus_cross_sample_batch":
        evidence_limit = min(CROSS_SAMPLE_EVIDENCE_LIMIT, max(max_train_samples, max_batches * 24, top_k * 24))
    evidence = _score_training_evidence(query_token, code_token, evidence_limit)
    candidate_indices: list[int] = []
    for item in evidence.get("supportingSamples", []) + evidence.get("conflictingSamples", []):
        train_index = int(item.get("trainIndex"))
        if train_index not in candidate_indices:
            candidate_indices.append(train_index)
        if loss_scope == "highlight_only" and len(candidate_indices) >= max_train_samples:
            break
        if loss_scope == "highlight_plus_cross_sample_batch" and len(candidate_indices) >= evidence_limit:
            break
    if not candidate_indices:
        return {
            "status": "cache_missing" if evidence.get("recordCount", 0) == 0 else "no_candidates",
            "source": "xsearch_step7000_gradient_influence_lite",
            "message": "No V2 training evidence candidates are available for gradient attribution.",
            "lossScope": loss_scope,
            "helpfulSamples": [],
            "harmfulSamples": [],
            "helpfulBatches": [],
            "harmfulBatches": [],
            "queryLoss": None,
        }

    pair_grad, pair_loss, pair_cosine = _pair_gradient(test_id, code_idx, query_token_index, code_token_index, mode)
    pair_norm = float(torch.norm(pair_grad).item())

    if loss_scope == "highlight_plus_cross_sample_batch":
        selected_batches = _select_candidate_batches(evidence, max_batches)

        scored_batches = []
        cache_hits = 0
        cache_misses = 0
        batch_selection_by_index = {int(item["batchIndex"]): item for item in selected_batches}
        for selected_batch in selected_batches:
            batch_index = int(selected_batch["batchIndex"])
            batch = _batch_gradient_cached(batch_index, cross_sample_weight)
            if batch is None:
                continue
            if batch.get("cacheHit"):
                cache_hits += 1
            else:
                cache_misses += 1
            grad = batch.pop("grad")
            dot = float(torch.dot(pair_grad, grad).item())
            batch_norm = float(torch.norm(grad).item())
            normalized = dot / max(pair_norm * batch_norm, 1e-12)
            influence = -dot
            batch.update(
                {
                    "influence": round(influence, 8),
                    "gradientDot": round(dot, 8),
                    "normalizedInfluence": round(-normalized, 8),
                    "trainGradientNorm": round(batch_norm, 8),
                    "interpretation": "helpful" if influence < 0 else "harmful",
                    "selection": batch_selection_by_index.get(batch_index, {}),
                }
            )
            scored_batches.append(batch)

        helpful_batches = sorted(
            [item for item in scored_batches if item["influence"] < 0],
            key=lambda item: item["influence"],
        )[:top_k]
        harmful_batches = sorted(
            [item for item in scored_batches if item["influence"] >= 0],
            key=lambda item: item["influence"],
            reverse=True,
        )[:top_k]
        return {
            "status": "ok",
            "source": "xsearch_step7000_gradient_influence_lite",
            "lossScope": loss_scope,
            "queryLoss": {
                "type": f"{mode}_loss",
                "formula": "-cos(q_token_hidden, c_token_hidden)" if mode == "pull" else "cos(q_token_hidden, c_token_hidden)",
                "value": round(pair_loss, 8),
                "modelCosine": round(pair_cosine, 8),
                "gradientNorm": round(pair_norm, 8),
            },
            "scope": {
                "parameterScope": list(GRADIENT_PARAM_NAMES),
                "lossScope": "highlight_plus_cross_sample_batch",
                "batchReconstruction": "sequential train batches, batch_size=32, batchIndex=trainIndex//32",
                "candidateSource": "full V2 training evidence retrieval",
                "candidateTrainSamples": len(candidate_indices),
                "candidateEvidenceLimit": evidence_limit,
                "evaluatedTrainSamples": sum(len(item.get("trainIndices", [])) for item in scored_batches),
                "evaluatedBatches": len(scored_batches),
                "requestedBatches": max_batches,
                "cacheHits": cache_hits,
                "cacheMisses": cache_misses,
                "crossSampleWeight": cross_sample_weight,
                "batchSelector": "selectionScore=maxEvidenceScore + 0.5*top3EvidenceScoreSum + 0.3*min(hitCount,5) + 0.2*min(exactTokenPairHits,3)",
                "influenceMethod": "gradient-dot-product approximation; no Hessian inverse",
            },
            "helpfulSamples": [],
            "harmfulSamples": [],
            "helpfulBatches": helpful_batches,
            "harmfulBatches": harmful_batches,
        }

    scored = []
    for train_index in candidate_indices:
        sample = _train_sample_gradient(train_index)
        if sample is None:
            continue
        grad = sample.pop("grad")
        dot = float(torch.dot(pair_grad, grad).item())
        train_norm = float(torch.norm(grad).item())
        normalized = dot / max(pair_norm * train_norm, 1e-12)
        influence = -dot
        sample.update(
            {
                "influence": round(influence, 8),
                "gradientDot": round(dot, 8),
                "normalizedInfluence": round(-normalized, 8),
                "trainGradientNorm": round(train_norm, 8),
                "interpretation": "helpful" if influence < 0 else "harmful",
            }
        )
        scored.append(sample)

    helpful = sorted([item for item in scored if item["influence"] < 0], key=lambda item: item["influence"])[:top_k]
    harmful = sorted([item for item in scored if item["influence"] >= 0], key=lambda item: item["influence"], reverse=True)[:top_k]
    return {
        "status": "ok",
        "source": "xsearch_step7000_gradient_influence_lite",
        "lossScope": loss_scope,
        "queryLoss": {
            "type": f"{mode}_loss",
            "formula": "-cos(q_token_hidden, c_token_hidden)" if mode == "pull" else "cos(q_token_hidden, c_token_hidden)",
            "value": round(pair_loss, 8),
            "modelCosine": round(pair_cosine, 8),
            "gradientNorm": round(pair_norm, 8),
        },
        "scope": {
            "parameterScope": list(GRADIENT_PARAM_NAMES),
            "lossScope": "highlight_only",
            "batchReconstruction": "sequential train batches, batch_size=32, batchIndex=trainIndex//32",
            "candidateSource": "full V2 training evidence retrieval",
            "evaluatedTrainSamples": len(scored),
            "evaluatedBatches": len({item["batchIndex"] for item in scored}),
            "influenceMethod": "gradient-dot-product approximation; no Hessian inverse",
        },
        "helpfulSamples": helpful,
        "harmfulSamples": harmful,
        "helpfulBatches": [],
        "harmfulBatches": [],
    }
