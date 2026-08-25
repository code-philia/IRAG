from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .config import API_BRIDGE_STEP7000_PACKED_PATH, ROOT_DIR, TRAINING_EVAL_RESULTS_DIR


PACKED_PREFIX = "python_full"
URL_INDEX_PATH = ROOT_DIR / "data" / "packed_url_index_python_full.json"
CODE_TOKEN_SLOT_OFFSET = 1


@dataclass(frozen=True)
class PackedStep:
    epoch: int
    path: Path
    label: str


@dataclass
class PackedTimeline:
    source: str
    kind: str
    epochs: list[int]
    code_vectors_by_epoch: list[np.ndarray]
    code_token_count: int
    hidden_dim: int
    url: str
    url_index: int
    token_slot_offset: int


def discover_packed_steps() -> list[PackedStep]:
    steps: list[PackedStep] = []
    if not TRAINING_EVAL_RESULTS_DIR.exists():
        return steps
    pattern = re.compile(rf"^{PACKED_PREFIX}_step(\d+)_packed\.pt$")
    for path in TRAINING_EVAL_RESULTS_DIR.glob(f"{PACKED_PREFIX}_step*_packed.pt"):
        match = pattern.match(path.name)
        if match:
            step = int(match.group(1))
            steps.append(PackedStep(epoch=step, path=path, label=f"step{step}"))
    steps.sort(key=lambda item: item.epoch)
    epoch1 = TRAINING_EVAL_RESULTS_DIR / f"{PACKED_PREFIX}_epoch1_packed.pt"
    if epoch1.exists():
        next_epoch = (steps[-1].epoch + 1000) if steps else 1
        steps.append(PackedStep(epoch=next_epoch, path=epoch1, label="epoch1"))
    return steps


def available_epochs() -> list[int]:
    return [step.epoch for step in discover_packed_steps()]


def _torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", mmap=True)


@lru_cache(maxsize=1)
def load_url_index() -> dict[str, int]:
    steps = discover_packed_steps()
    if not steps:
        return {}
    _hidden, _mask, urls = _torch_load(steps[0].path)
    url_index = {str(url): idx for idx, url in enumerate(urls)}
    return url_index


def get_packed_timeline(url: str, requested_code_tokens: int) -> PackedTimeline | None:
    steps = discover_packed_steps()
    if not steps or not url:
        return None

    url_index = load_url_index()
    if url not in url_index:
        if not API_BRIDGE_STEP7000_PACKED_PATH.exists():
            return None
        hidden, _mask, urls = _torch_load(API_BRIDGE_STEP7000_PACKED_PATH)
        bridge_index = {str(item): idx for idx, item in enumerate(urls)}
        if url not in bridge_index:
            return None
        token_count = max(0, min(int(requested_code_tokens), 64 - CODE_TOKEN_SLOT_OFFSET))
        if token_count == 0:
            return None
        sample = hidden[bridge_index[url], CODE_TOKEN_SLOT_OFFSET : CODE_TOKEN_SLOT_OFFSET + token_count].detach().cpu()
        vectors = sample.numpy().astype(np.float32, copy=True)
        return PackedTimeline(
            source="api_bridge_step7000_subset_cache",
            kind="last_layer_code_token_hidden",
            epochs=[1, 2, 3, 4],
            code_vectors_by_epoch=[vectors, vectors.copy(), vectors.copy(), vectors.copy()],
            code_token_count=token_count,
            hidden_dim=int(vectors.shape[1]),
            url=url,
            url_index=int(bridge_index[url]),
            token_slot_offset=CODE_TOKEN_SLOT_OFFSET,
        )

    row_index = url_index[url]
    token_count = max(0, min(int(requested_code_tokens), 64 - CODE_TOKEN_SLOT_OFFSET))
    if token_count == 0:
        return None

    epochs: list[int] = []
    vectors_by_epoch: list[np.ndarray] = []
    hidden_dim = 0
    for step in steps:
        hidden, _mask, _urls = _torch_load(step.path)
        sample = hidden[row_index, CODE_TOKEN_SLOT_OFFSET : CODE_TOKEN_SLOT_OFFSET + token_count].detach().cpu()
        arr = sample.to(dtype=sample.dtype).numpy().astype(np.float32, copy=True)
        if arr.ndim != 2:
            continue
        hidden_dim = int(arr.shape[1])
        epochs.append(step.epoch)
        vectors_by_epoch.append(arr)

    if not vectors_by_epoch:
        return None

    return PackedTimeline(
        source="packed_cache",
        kind="last_layer_code_token_hidden",
        epochs=epochs,
        code_vectors_by_epoch=vectors_by_epoch,
        code_token_count=token_count,
        hidden_dim=hidden_dim,
        url=url,
        url_index=row_index,
        token_slot_offset=CODE_TOKEN_SLOT_OFFSET,
    )


def fallback_metadata(reason: str) -> dict[str, Any]:
    return {
        "representationSource": "fallback",
        "representationKind": "synthetic_token_features",
        "fallbackReason": reason,
        "availableEpochs": [1, 2, 3, 4],
    }
