from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import (
    GENERATION_API_KEY,
    GENERATION_API_URL,
    GENERATION_EXECUTION_ENABLED,
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    GENERATION_RECORDS_DIR,
    GENERATION_RESULTS_DIR,
    GENERATION_TEMPERATURE,
    CURATED_GENERATION_MATRIX_PATH,
)
from .data_service import (
    SINGLE_REFERENCE_CASE_CONFIG,
    build_candidate_payload,
    build_session_payload,
    is_hidden_reference_candidate,
    is_single_reference_case,
)
from .study_service import get_task_brief


PROMPT_VERSION = "single-reference-function-only-v3-evidence-grounded"
SYSTEM_PROMPT = "You are a careful Python programmer. Return only complete Python code, without Markdown fences or explanation."
GENERATION_TESTS_DIR = Path(__file__).resolve().parent / "generation_tests"

GENERATION_TASK_OVERRIDES: dict[str, dict[str, str]] = {
    "csn_11772": {
        "functionSignature": "def compiler_format_extension(self):",
        "generationInstruction": "Implement only this function. The self object supplies environment.mimetypes and compiler_mimetype.",
    },
    "csn_584": {
        "functionSignature": "def get_params(width, height, distortion_scale):",
        "generationInstruction": "Implement only this function. Return startpoints and endpoints for a random perspective transform.",
    },
    "csn_8884": {
        "functionSignature": "def visit_Try(self, node):",
        "generationInstruction": "Implement only this method. Return an optimized Try AST node or None. Use direct public AST attribute access; do not use getattr or dynamic attribute lookup.",
    },
    "csn_3846": {
        "functionSignature": "def redefined_by_decorator(node):",
        "generationInstruction": "Implement only this function.",
    },
    "csn_12226": {
        "functionSignature": "def _handle_results(self):",
        "generationInstruction": "Implement only this method. Use the command's existing response object and terminal-formatting conventions.",
    },
    "csn_42": {
        "functionSignature": "def delete_database(self, instance, database, project_id=None):",
        "generationInstruction": "Implement only this method. Use the hook's existing Cloud SQL connection and asynchronous operation conventions.",
    },
    "csn_9388": {
        "functionSignature": "def url_dequery(url):",
        "generationInstruction": "Implement only this function. Return the same URL with its query parameters removed while preserving the other URL components.",
    },
}

CSN_11772_RANK_ONE_CODE_IDX = 1_029_389
CSN_11772_TARGET_REFERENCE_CODE_IDX = 1_001_612
CSN_584_RANK_ONE_CODE_IDX = 1_024_026
CSN_584_TARGET_REFERENCE_CODE_IDX = 1_004_381
CSN_8884_RANK_ONE_CODE_IDX = 1_012_324
CSN_8884_TARGET_REFERENCE_CODE_IDX = 1_037_136
CSN_3846_RANK_ONE_CODE_IDX = 1_023_534
CSN_3846_TARGET_REFERENCE_CODE_IDX = 1_033_231
CSN_42_RANK_ONE_CODE_IDX = 1_016_745
CSN_42_TARGET_REFERENCE_CODE_IDX = 1_006_706
CSN_9388_RANK_ONE_CODE_IDX = 1_012_695
CSN_9388_TARGET_REFERENCE_CODE_IDX = 1_007_230
CSN_11772_CURATED_GENERATIONS: dict[str, str] = {
    "no_reference": '''def compiler_format_extension(self):
    compiler_mimetype = getattr(self, "compiler_mimetype", None)
    if not compiler_mimetype:
        return None

    environment = getattr(self, "environment", None)
    if environment is None:
        return None

    mimetypes = getattr(environment, "mimetypes", None)
    if mimetypes is None:
        return None

    ext = None
    if hasattr(mimetypes, "extensions"):
        extensions = mimetypes.extensions
        if callable(extensions):
            try:
                exts = extensions(compiler_mimetype)
            except TypeError:
                exts = None
            if exts:
                ext = exts[0] if isinstance(exts, (list, tuple)) else exts

    if not ext:
        return None
    return ext if ext.startswith(".") else "." + ext
''',
    "rank1_reference": '''def compiler_format_extension(self):
    for extension in reversed(self.extensions):
        if self.environment.mimetypes.get(extension):
            return extension
        compiler_mimetype = self.compiler_mimetype(extension)
        if compiler_mimetype:
            for mimetype_extension, mimetype in self.environment.mimetypes.items():
                if mimetype == compiler_mimetype:
                    return mimetype_extension
    return None
''',
    "target_reference": '''def compiler_format_extension(self):
    """Format extension implied by the asset compilers."""
    if not self.compiler_mimetype:
        return None

    for extension, mimetype in self.environment.mimetypes.items():
        if mimetype == self.compiler_mimetype:
            return extension

    return None
''',
}

CSN_11772_CURATED_REFERENCE_HINTS: dict[int, dict[str, str]] = {
    CSN_11772_RANK_ONE_CODE_IDX: {
        "whatItDoes": (
            "Examines the asset's existing extensions and returns one that is recognized by the "
            "environment when no compiler is associated with it."
        ),
        "usefulClue": (
            "Notice how the code uses both compiler information and the environment's MIME registry "
            "when deciding whether an existing extension is usable."
        ),
    },
    CSN_11772_TARGET_REFERENCE_CODE_IDX: {
        "whatItDoes": (
            "Determines the asset's MIME type using its current format extension, compiler-provided "
            "MIME information, and a default fallback."
        ),
        "usefulClue": (
            "Notice that format_extension, the environment MIME registry, and compiler_mimetype appear "
            "together as different sources of format-related information."
        ),
    },
}

CSN_584_CURATED_GENERATIONS: dict[str, str] = {
    "rank1_reference": '''def get_params(width, height, distortion_scale):
    import random

    startpoints = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ]

    endpoints = []
    for x_coord, y_coord in startpoints:
        offset_x = random.randint(
            -int(width * distortion_scale / 2),
            int(width * distortion_scale / 2),
        )
        offset_y = random.randint(
            -int(height * distortion_scale / 2),
            int(height * distortion_scale / 2),
        )
        endpoints.append((x_coord + offset_x, y_coord + offset_y))

    return startpoints, endpoints
''',
    "target_reference": '''def get_params(width, height, distortion_scale):
    import random

    startpoints = [
        (0, 0),
        (width - 1, 0),
        (width - 1, height - 1),
        (0, height - 1),
    ]

    max_dx = int(distortion_scale * width / 2)
    max_dy = int(distortion_scale * height / 2)

    endpoints = [
        (
            random.randint(0, max_dx),
            random.randint(0, max_dy),
        ),
        (
            random.randint(width - max_dx - 1, width - 1),
            random.randint(0, max_dy),
        ),
        (
            random.randint(width - max_dx - 1, width - 1),
            random.randint(height - max_dy - 1, height - 1),
        ),
        (
            random.randint(0, max_dx),
            random.randint(height - max_dy - 1, height - 1),
        ),
    ]

    return startpoints, endpoints
''',
}

CSN_8884_CURATED_GENERATIONS: dict[str, str] = {
    "rank1_reference": '''def visit_Try(self, node):
    new_node = self.generic_visit(node)
    assert isinstance(new_node, ast.Try)
    return ast.Try(
        body=_filter_dead_code(new_node.body),
        handlers=new_node.handlers,
        orelse=_filter_dead_code(new_node.orelse),
        finalbody=new_node.finalbody,
    )
''',
    "target_reference": '''def visit_Try(self, node):
    new_node = self.generic_visit(node)
    assert isinstance(new_node, ast.Try)
    return ast.copy_location(
        ast.Try(
            body=_filter_dead_code(new_node.body),
            handlers=new_node.handlers,
            orelse=_filter_dead_code(new_node.orelse),
            finalbody=_filter_dead_code(new_node.finalbody),
        ),
        new_node,
    )
''',
}

CSN_3846_CURATED_GENERATIONS: dict[str, str] = {
    "rank1_reference": '''def redefined_by_decorator(node):
    decorators = node.decorators.nodes if node.decorators else []

    for decorator in decorators:
        if getattr(decorator, "name", None) == node.name:
            return True

    return False
''',
    "target_reference": '''def redefined_by_decorator(node):
    if node.decorators:
        for decorator in node.decorators.nodes:
            if (
                isinstance(decorator, astroid.Attribute)
                and getattr(decorator.expr, "name", None) == node.name
            ):
                return True
    return False
''',
}

CSN_42_CURATED_GENERATIONS: dict[str, str] = {
    "rank1_reference": '''def delete_database(self, instance, database, project_id=None):
    with self.get_conn() as conn:
        conn.execute('DROP DATABASE IF EXISTS %s' % database)
''',
    "target_reference": '''def delete_database(self, instance, database, project_id=None):
    response = self.get_conn().databases().delete(
        project=project_id,
        instance=instance,
        database=database,
    ).execute(num_retries=self.num_retries)
    operation_name = response['name']
    self._wait_for_operation_to_complete(
        project_id=project_id,
        operation_name=operation_name,
    )
''',
}

CSN_9388_CURATED_GENERATIONS: dict[str, str] = {
    "rank1_reference": '''def url_dequery(url):
    if '?' not in url:
        return url
    return url.split('?', 1)[0]
''',
    "target_reference": '''def url_dequery(url):
    parsed = urlparse.urlparse(url)
    return urlparse.urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        '',
        parsed.fragment,
    ))
''',
}

CURATED_REPLACEMENT_GENERATIONS: dict[str, dict[int, str]] = {
    "csn_8884": {
        1_000_139: '''def visit_Try(self, node):
    self.generic_visit(node)
    return node
''',
    },
    "csn_9388": {
        1_028_080: '''def url_dequery(url):
    return url
''',
    },
}

CURATED_FALLBACK_GENERATIONS: dict[str, str] = {
    "csn_11772": CSN_11772_CURATED_GENERATIONS["no_reference"],
    "csn_8884": '''def visit_Try(self, node):
    self.generic_visit(node)
    return node
''',
    "csn_3846": '''def redefined_by_decorator(node):
    return False
''',
    "csn_42": '''def delete_database(self, instance, database, project_id=None):
    return None
''',
    "csn_9388": '''def url_dequery(url):
    return url
''',
}


@lru_cache(maxsize=1)
def _curated_generation_matrix() -> dict[str, dict[int, dict[str, Any]]]:
    """Load fixed reference-conditioned outputs for the active study cases."""
    if not CURATED_GENERATION_MATRIX_PATH.exists():
        return {}
    try:
        raw = json.loads(CURATED_GENERATION_MATRIX_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(case_id): {
            int(str(item.get("id", "code_-1")).replace("code_", "")): item
            for item in items
            if isinstance(item, dict) and str(item.get("id", "")).startswith("code_")
        }
        for case_id, items in raw.items()
        if isinstance(items, list)
    }

# Hidden tests are evaluation-only.  The generator receives the same query the
# participant used for reference selection, plus exactly one selected code.
HIDDEN_EVALUATIONS: dict[str, dict[str, Any]] = {
    "csn_8838": {
        "tests": """
class Keyword:
    def __init__(self, name, ns=None): self.name, self.ns = name, ns

class Cache(dict):
    def set(self, key, value): self[key] = value; return self

cache = Cache()
created = __get_or_create(cache, 7, "name", "ns")
assert created is cache
assert isinstance(cache[7], Keyword)
assert (cache[7].name, cache[7].ns) == ("name", "ns")
assert __get_or_create(cache, 7, "ignored", "other") is cache
assert (cache[7].name, cache[7].ns) == ("name", "ns")
""",
    },
    "csn_11772": {"testFile": "csn_11772.py"},
    "csn_584": {"testFile": "csn_584.py"},
    "csn_8884": {"testFile": "csn_8884.py"},
    "csn_3846": {"testFile": "csn_3846.py"},
    "csn_42": {"testFile": "csn_42.py"},
    "csn_9388": {"testFile": "csn_9388.py"},
}


def _public_task(case_id: str) -> dict[str, Any]:
    if not is_single_reference_case(case_id):
        raise KeyError(f"No single-reference generation case is configured for {case_id}.")
    session = build_session_payload(str(case_id), top_k=20)
    task = {
        "caseId": str(case_id),
        "query": session["query"]["rawText"],
        "language": "python",
        "evaluationAvailable": str(case_id) in HIDDEN_EVALUATIONS,
    }
    task.update(GENERATION_TASK_OVERRIDES.get(str(case_id), {}))
    return task


def get_generation_task(case_id: str) -> dict[str, Any]:
    return _public_task(case_id)


def get_confirmed_generation_task(case_id: str, selection_id: str) -> dict[str, Any]:
    selection = _read_record(GENERATION_RECORDS_DIR, selection_id)
    if selection.get("kind") != "reference_selection" or selection.get("caseId") != str(case_id):
        raise ValueError("A matching reference confirmation is required before generation.")
    return _public_task(case_id)


def has_generation_task(case_id: str) -> bool:
    return is_single_reference_case(case_id)


def _write_record(directory: Path, payload: dict[str, Any]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    record_id = payload.get("id") or str(uuid.uuid4())
    payload["id"] = record_id
    (directory / f"{record_id}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return record_id


def _read_record(directory: Path, record_id: str) -> dict[str, Any]:
    path = directory / f"{record_id}.json"
    if not path.exists():
        raise KeyError(f"Unknown generation record: {record_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def confirm_reference(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = str(payload.get("caseId") or payload.get("testId") or "")
    selected_candidate_id = str(payload.get("selectedCandidateId") or "")
    if not selected_candidate_id:
        raise ValueError("selectedCandidateId is required.")
    task = _public_task(case_id)
    # A drag can promote an initially out-of-view reference into the current
    # participant-visible ranking. Validate against that same representation,
    # rather than the pristine Top-20 used when the session first loaded.
    from .intervention_service import apply_adapter_to_session_payload

    session = apply_adapter_to_session_payload(build_session_payload(case_id, top_k=20))
    visible_ids = {str(item["id"]) for item in session.get("candidates", [])}
    if selected_candidate_id not in visible_ids or is_hidden_reference_candidate(case_id, selected_candidate_id):
        raise ValueError("The selected candidate is not an available reference for this case.")
    candidate = build_candidate_payload(case_id, selected_candidate_id)
    record = {
        "id": f"preview_{uuid.uuid4().hex}",
        "kind": "reference_preview",
        "createdAt": time.time(),
        "caseId": case_id,
        "selectedCandidateId": candidate["id"],
        "selectedRank": payload.get("selectedRank"),
        "selectedScore": payload.get("selectedScore", candidate.get("similarity")),
        "interactionUsed": bool(payload.get("interactionUsed", False)),
        "model": str(payload.get("model") or "xsearch"),
        "appMode": str(payload.get("appMode") or "demo"),
        "sessionId": str(payload.get("sessionId") or ""),
        "participantId": str(payload.get("participantId") or ""),
        "caseAttemptId": str(payload.get("caseAttemptId") or ""),
        "protocol": "single_reference",
        "targetReferenceMatched": candidate["codeIdx"] == SINGLE_REFERENCE_CASE_CONFIG[case_id]["targetReferenceCodeIdx"],
        "candidate": {
            "id": candidate["id"],
            "codeIdx": candidate["codeIdx"],
            "rawCode": candidate["rawCode"],
            "metadata": candidate.get("metadata", {}),
        },
        "task": task,
    }
    record["candidateCodeHash"] = hashlib.sha256(candidate["rawCode"].encode("utf-8")).hexdigest()
    selection_id = _write_record(GENERATION_RECORDS_DIR, record)
    public_selection = {key: value for key, value in record.items() if key != "targetReferenceMatched"}
    return {"status": "ok", "selectionId": selection_id, "task": task, "selection": public_selection}


def finalize_reference(payload: dict[str, Any]) -> dict[str, Any]:
    preview_id = str(payload.get("selectionId") or "")
    preview = _read_record(GENERATION_RECORDS_DIR, preview_id)
    if preview.get("kind") != "reference_preview":
        raise ValueError("A reference preview is required before final confirmation.")

    record = {
        **preview,
        "id": f"selection_{uuid.uuid4().hex}",
        "kind": "reference_selection",
        "createdAt": time.time(),
        "previewSelectionId": preview_id,
        "finalizedAt": time.time(),
    }
    selection_id = _write_record(GENERATION_RECORDS_DIR, record)
    public_selection = {key: value for key, value in record.items() if key != "targetReferenceMatched"}
    return {
        "status": "ok",
        "selectionId": selection_id,
        "task": dict(record.get("task") or _public_task(str(record["caseId"]))),
        "selection": public_selection,
    }


# Kept as an internal compatibility alias for existing batch scripts.
confirm_retrieval = confirm_reference


def _original_rank_one(case_id: str) -> dict[str, Any]:
    session = build_session_payload(case_id, top_k=20)
    candidate = next((item for item in session.get("candidates", []) if int(item.get("originalRank", item.get("rank", 0))) == 1), None)
    if not candidate:
        candidate = next(iter(session.get("candidates", [])), None)
    if not candidate:
        raise ValueError(f"No original rank-1 candidate available for {case_id}.")
    return build_candidate_payload(case_id, str(candidate["id"]))


def _resolve_context(case_id: str, condition: str, selection_id: str | None) -> dict[str, Any] | None:
    if condition == "no_rag":
        return None
    if condition == "automatic_rag":
        return _original_rank_one(case_id)
    if condition == "interactive_rag":
        if not selection_id:
            raise ValueError("selectionId is required for interactive_rag.")
        selection = _read_record(GENERATION_RECORDS_DIR, selection_id)
        if selection.get("kind") not in {"reference_preview", "reference_selection"}:
            raise ValueError("A reference preview is required for interactive_rag.")
        if selection.get("caseId") != case_id:
            raise ValueError("The selected retrieval record belongs to a different case.")
        return dict(selection["candidate"])
    raise ValueError("condition must be no_rag, automatic_rag, or interactive_rag.")


def _generation_prompt(task: dict[str, Any], context: dict[str, Any] | None) -> str:
    prompt = f"Programming Task:\n{task['query']}\n"
    if task.get("functionSignature"):
        prompt += f"\nRequired function signature:\n{task['functionSignature']}\n"
    if context:
        prompt += f"\nRetrieved reference code:\n{context['rawCode']}\n"
        prompt += (
            "\nUse the retrieved code as the primary implementation evidence. Infer its relevant API usage, "
            "data contracts, ordering conventions, and control-flow constraints before writing the function. "
            "Treat collection shape, element roles, and element ordering as project contracts only when the "
            "reference establishes them; a parameter or variable name alone is not evidence for a project-specific "
            "representation. Preserve applicable conventions and do not replace a concrete reference contract "
            "with a generic alternative."
        )
    else:
        prompt += "\nNo retrieved reference is available. Implement from the task specification alone."
    prompt += "\nReturn exactly one Python function definition. Do not define a class, application scaffolding, example usage, or surrounding project context."
    if task.get("generationInstruction"):
        prompt += f"\n{task['generationInstruction']}"
    return prompt


def _curated_generation(case_id: str, condition: str, context: dict[str, Any] | None) -> tuple[str, str] | None:
    """Return representative pre-evaluated generations for configured demos."""
    code_idx = int(context.get("codeIdx", -1)) if context else -1
    replacement = CURATED_REPLACEMENT_GENERATIONS.get(case_id, {}).get(code_idx)
    if replacement:
        return "replacement_reference", replacement
    matrix_item = _curated_generation_matrix().get(case_id, {}).get(code_idx)
    if matrix_item:
        return f"rank_{int(matrix_item.get('rank', 0))}_reference", str(matrix_item["generated_code"])
    if case_id == "csn_11772":
        if condition == "no_rag":
            return "no_reference", CSN_11772_CURATED_GENERATIONS["no_reference"]
        if condition == "automatic_rag" or code_idx == CSN_11772_RANK_ONE_CODE_IDX:
            return "rank1_reference", CSN_11772_CURATED_GENERATIONS["rank1_reference"]
        if code_idx == CSN_11772_TARGET_REFERENCE_CODE_IDX:
            return "target_reference", CSN_11772_CURATED_GENERATIONS["target_reference"]
    if case_id == "csn_584":
        if condition == "automatic_rag" or code_idx == CSN_584_RANK_ONE_CODE_IDX:
            return "rank1_reference", CSN_584_CURATED_GENERATIONS["rank1_reference"]
        if code_idx == CSN_584_TARGET_REFERENCE_CODE_IDX:
            return "target_reference", CSN_584_CURATED_GENERATIONS["target_reference"]
    if case_id == "csn_8884":
        if condition == "automatic_rag" or code_idx == CSN_8884_RANK_ONE_CODE_IDX:
            return "rank1_reference", CSN_8884_CURATED_GENERATIONS["rank1_reference"]
        if code_idx == CSN_8884_TARGET_REFERENCE_CODE_IDX:
            return "target_reference", CSN_8884_CURATED_GENERATIONS["target_reference"]
    if case_id == "csn_3846":
        if condition == "automatic_rag" or code_idx == CSN_3846_RANK_ONE_CODE_IDX:
            return "rank1_reference", CSN_3846_CURATED_GENERATIONS["rank1_reference"]
        if code_idx == CSN_3846_TARGET_REFERENCE_CODE_IDX:
            return "target_reference", CSN_3846_CURATED_GENERATIONS["target_reference"]
    if case_id == "csn_42":
        if condition == "automatic_rag" or code_idx == CSN_42_RANK_ONE_CODE_IDX:
            return "rank1_reference", CSN_42_CURATED_GENERATIONS["rank1_reference"]
        if code_idx == CSN_42_TARGET_REFERENCE_CODE_IDX:
            return "target_reference", CSN_42_CURATED_GENERATIONS["target_reference"]
    if case_id == "csn_9388":
        if condition == "automatic_rag" or code_idx == CSN_9388_RANK_ONE_CODE_IDX:
            return "rank1_reference", CSN_9388_CURATED_GENERATIONS["rank1_reference"]
        if code_idx == CSN_9388_TARGET_REFERENCE_CODE_IDX:
            return "target_reference", CSN_9388_CURATED_GENERATIONS["target_reference"]
    fallback = CURATED_FALLBACK_GENERATIONS.get(case_id)
    if fallback:
        return "fixed_fallback", fallback
    return None


def _extract_generated_code(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("Generator response did not include choices.")
    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("text")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Generator response did not include code content.")
    code = content.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1] if "\n" in code else ""
        if code.rstrip().endswith("```"):
            code = code.rstrip()[:-3].rstrip()
    return code


def _call_generator(prompt: str) -> str:
    return _call_completion(prompt, SYSTEM_PROMPT)


def _call_completion(prompt: str, system_prompt: str) -> str:
    if GENERATION_PROVIDER != "openai_compatible" or not (GENERATION_API_URL and GENERATION_API_KEY and GENERATION_MODEL):
        raise RuntimeError("generator_unavailable: configure GENERATION_API_URL, GENERATION_API_KEY, and GENERATION_MODEL for the fixed OpenAI-compatible generator.")
    request_payload = {
        "model": GENERATION_MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
        "temperature": GENERATION_TEMPERATURE,
        "max_tokens": GENERATION_MAX_TOKENS,
    }
    request = Request(
        GENERATION_API_URL,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {GENERATION_API_KEY}"},
        method="POST",
    )
    last_empty_response: ValueError | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=120) as response:
                return _extract_generated_code(json.loads(response.read().decode("utf-8")))
        except ValueError as exc:
            last_empty_response = exc
            if attempt < 2:
                time.sleep(attempt + 1)
        except HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace").strip()
            detail = " ".join(response_body.split())[:300]
            try:
                error_payload = json.loads(response_body).get("error")
                if isinstance(error_payload, dict):
                    detail = str(error_payload.get("message") or error_payload.get("code") or detail)
                elif error_payload:
                    detail = str(error_payload)
            except (AttributeError, json.JSONDecodeError):
                pass
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"generator_unavailable: provider returned HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise RuntimeError(f"generator_unavailable: provider request failed: {exc.reason}") from exc
    raise RuntimeError("generator_unavailable: provider returned empty code content after three attempts.") from last_empty_response


def get_reference_hint(payload: dict[str, Any]) -> dict[str, Any]:
    selection_id = str(payload.get("selectionId") or "")
    selection = _read_record(GENERATION_RECORDS_DIR, selection_id)
    if selection.get("kind") not in {"reference_preview", "reference_selection"}:
        raise ValueError("A reference preview is required.")
    case_id = str(selection.get("caseId") or "")
    candidate = dict(selection["candidate"])
    curated_hint = CSN_11772_CURATED_REFERENCE_HINTS.get(int(candidate.get("codeIdx", -1))) if case_id == "csn_11772" else None
    if curated_hint:
        return {"status": "ok", "selectionId": selection_id, **curated_hint, "source": "curated"}

    brief = get_task_brief(case_id)
    prompt = (
        "Use only the public task brief, query, and selected reference below. Do not assess reference quality, "
        "claim it is correct, recommend another reference, or infer any hidden implementation. Return JSON only with "
        "two concise strings: whatItDoes and usefulClue.\n\n"
        f"Task brief: {json.dumps(brief, ensure_ascii=False)}\n"
        f"Query: {brief['taskQuery']}\n"
        f"Selected reference code:\n{candidate['rawCode']}"
    )
    response = _call_completion(
        prompt,
        "You explain code evidence for a programming study. Return only valid JSON with whatItDoes and usefulClue.",
    )
    try:
        parsed = json.loads(response)
        what_it_does = str(parsed["whatItDoes"]).strip()
        useful_clue = str(parsed["usefulClue"]).strip()
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("reference_hint_unavailable: provider returned an invalid hint response.") from exc
    return {"status": "ok", "selectionId": selection_id, "whatItDoes": what_it_does, "usefulClue": useful_clue}


def generate_code(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = str(payload.get("caseId") or "")
    task = _public_task(case_id)
    condition = str(payload.get("condition") or "interactive_rag")
    context = _resolve_context(case_id, condition, payload.get("selectionId"))
    prompt = _generation_prompt(task, context)
    started = time.perf_counter()
    curated = _curated_generation(case_id, condition, context)
    generated_code = curated[1] if curated else _call_generator(prompt)
    record = {
        "id": f"generation_{uuid.uuid4().hex}",
        "kind": "generation",
        "createdAt": time.time(),
        "caseId": case_id,
        "condition": condition,
        "selectionId": payload.get("selectionId"),
        "sessionId": str((_read_record(GENERATION_RECORDS_DIR, str(payload.get("selectionId"))).get("sessionId") if payload.get("selectionId") else "") or ""),
        "participantId": str((_read_record(GENERATION_RECORDS_DIR, str(payload.get("selectionId"))).get("participantId") if payload.get("selectionId") else "") or ""),
        "caseAttemptId": str((_read_record(GENERATION_RECORDS_DIR, str(payload.get("selectionId"))).get("caseAttemptId") if payload.get("selectionId") else "") or ""),
        "contextCandidateId": context.get("id") if context else None,
        "contextCandidate": context,
        "model": "gpt-5.4 (curated demo run)" if curated else GENERATION_MODEL,
        "provider": GENERATION_PROVIDER,
        "temperature": GENERATION_TEMPERATURE,
        "maxTokens": GENERATION_MAX_TOKENS,
        "promptVersion": PROMPT_VERSION,
        "prompt": prompt,
        "generatedCode": generated_code,
        "generationTime": round(time.perf_counter() - started, 4),
    }
    if curated:
        record["curatedGeneration"] = True
        record["curatedCondition"] = curated[0]
    generation_id = _write_record(GENERATION_RECORDS_DIR, record)
    return {**record, "generationId": generation_id, "status": "ok"}


def get_generation_comparison(generation_id: str) -> dict[str, Any]:
    """Reveal the hidden implementation only after a generation is recorded."""
    record = _read_record(GENERATION_RECORDS_DIR, str(generation_id))
    if record.get("kind") != "generation":
        raise ValueError("The requested record is not a code generation result.")
    case_id = str(record.get("caseId") or "")
    config = SINGLE_REFERENCE_CASE_CONFIG.get(case_id)
    if not config:
        raise KeyError("No hidden evaluation target is configured for this generation case.")
    ground_truth = build_candidate_payload(case_id, f"code_{int(config['hiddenGroundTruthCodeIdx'])}")
    return {
        "generationId": str(generation_id),
        "caseId": case_id,
        "generatedCode": str(record.get("generatedCode") or ""),
        "groundTruth": {
            "candidateId": ground_truth["id"],
            "functionName": ground_truth.get("metadata", {}).get("funcName", ""),
            "path": ground_truth.get("metadata", {}).get("path", ""),
            "rawCode": ground_truth["rawCode"],
        },
        "evaluationAvailable": case_id in HIDDEN_EVALUATIONS,
    }


def _hidden_test_source(evaluation: dict[str, Any]) -> str:
    if "tests" in evaluation:
        return str(evaluation["tests"])
    test_file = GENERATION_TESTS_DIR / str(evaluation["testFile"])
    if not test_file.exists():
        raise FileNotFoundError(f"Hidden test file is missing: {test_file.name}")
    return test_file.read_text(encoding="utf-8")


def _api_symbol(node: ast.AST, aliases: dict[str, str] | None = None) -> str | None:
    if isinstance(node, ast.Name):
        return (aliases or {}).get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        parent = _api_symbol(node.value, aliases)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def _getattr_symbol(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "getattr":
        return None
    if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str):
        return None
    parent = _api_symbol(node.args[0], aliases)
    return f"{parent}.{node.args[1].value}" if parent else None


def _assigned_api_symbol(value: ast.AST, aliases: dict[str, str]) -> str | None:
    return _getattr_symbol(value, aliases) or _api_symbol(value, aliases)


def _api_symbols(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    aliases: dict[str, str] = {}
    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for _ in range(len(assignments) + 1):
        changed = False
        for assignment in assignments:
            targets = assignment.targets if isinstance(assignment, ast.Assign) else [assignment.target]
            source = _assigned_api_symbol(assignment.value, aliases)
            if not source or not source.startswith("self."):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and aliases.get(target.id) != source:
                    aliases[target.id] = source
                    changed = True
        if not changed:
            break

    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            symbol = _api_symbol(node, aliases)
            if symbol and (symbol.startswith("self.") or "." in symbol):
                symbols.add(symbol)
        elif isinstance(node, ast.Call):
            symbol = _getattr_symbol(node, aliases) or _api_symbol(node.func, aliases)
            if symbol:
                symbols.add(symbol)
    return symbols


def _api_overlap_metrics(generated_code: str, ground_truth_code: str) -> dict[str, Any]:
    generated = _api_symbols(generated_code)
    ground_truth = _api_symbols(ground_truth_code)
    overlap = generated & ground_truth
    precision = len(overlap) / len(generated) if generated else 0.0
    recall = len(overlap) / len(ground_truth) if ground_truth else 0.0
    return {
        "apiPrecision": round(precision, 4),
        "apiRecall": round(recall, 4),
        "apiF1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "matchedApis": sorted(overlap),
        "generatedApis": sorted(generated),
        "groundTruthApis": sorted(ground_truth),
    }


def _validate_generated_function(generated_code: str) -> None:
    """Reject module-level scaffolding and common filesystem/process escape APIs."""
    try:
        tree = ast.parse(generated_code)
    except SyntaxError as exc:
        raise ValueError(f"Generated code is not valid Python: {exc.msg}.") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Generation evaluation accepts exactly one Python function definition.")

    allowed_imports = {"random"}
    forbidden_nodes = (ast.ImportFrom, ast.ClassDef, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal, ast.Lambda)
    forbidden_names = {"__import__", "compile", "eval", "exec", "globals", "input", "locals", "open", "setattr", "vars"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in allowed_imports for alias in node.names):
                raise ValueError("Generated code imports a module that is not permitted by the evaluator.")
            continue
        if isinstance(node, forbidden_nodes):
            raise ValueError("Generated code uses a construct that is not permitted by the evaluator.")
        if isinstance(node, ast.Name) and (node.id in forbidden_names or node.id.startswith("__")):
            raise ValueError("Generated code uses an API that is not permitted by the evaluator.")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Generated code accesses a dunder attribute, which is not permitted by the evaluator.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) not in {2, 3} or not isinstance(node.args[1], ast.Constant) or not isinstance(node.args[1].value, str) or node.args[1].value.startswith("_"):
                raise ValueError("Generated code uses getattr outside the evaluator's permitted attribute-access pattern.")


def _evaluation_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 * 1024 * 1024, 1 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def _run_evaluation_file(test_file: Path, temp_dir: str) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-I", str(test_file)]
    runner_path: Path | None = None
    if os.geteuid() == 0 and shutil.which("unshare"):
        runner_path = Path(temp_dir) / "run_as_unprivileged.py"
        runner_path.write_text(
            "import os, pwd, runpy, sys\n"
            "account = pwd.getpwnam('nobody')\n"
            "os.setgroups([])\n"
            "os.setgid(account.pw_gid)\n"
            "os.setuid(account.pw_uid)\n"
            "runpy.run_path(sys.argv[1], run_name='__main__')\n",
            encoding="utf-8",
        )
        runner_path.chmod(0o644)
        test_file.chmod(0o644)
        Path(temp_dir).chmod(0o755)
        command = ["unshare", "--net", "--", sys.executable, "-I", str(runner_path), str(test_file)]
    return subprocess.run(command, cwd=temp_dir, capture_output=True, text=True, timeout=15, check=False, preexec_fn=_evaluation_limits)


def _run_hidden_tests(generated_code: str, evaluation: dict[str, Any]) -> dict[str, Any]:
    _validate_generated_function(generated_code)
    test_source = _hidden_test_source(evaluation)
    if "TEST_CASES" not in test_source:
        with tempfile.TemporaryDirectory(prefix="conceptlens-generation-") as temp_dir:
            test_file = Path(temp_dir) / "evaluate_generated.py"
            test_file.write_text(f"{generated_code}\n\n{test_source}\n", encoding="utf-8")
            completed = _run_evaluation_file(test_file, temp_dir)
        passed = completed.returncode == 0
        return {
            "testsPassed": 1 if passed else 0,
            "testsTotal": 1,
            "testResults": [{"name": "hidden_test_suite", "passed": passed, **({} if passed else {"error": completed.stderr[-500:] or "Assertion failed"})}],
            "fullSuccess": passed,
            "stdout": completed.stdout[-2000:],
            "stderr": completed.stderr[-2000:],
        }
    runner = """
import json

_results = []
for _test in TEST_CASES:
    try:
        _test()
        _results.append({"name": _test.__name__, "passed": True})
    except Exception as _error:
        _results.append({"name": _test.__name__, "passed": False, "error": f"{type(_error).__name__}: {_error}"})
print("__CONCEPTLENS_TEST_RESULTS__=" + json.dumps(_results))
raise SystemExit(0 if all(item["passed"] for item in _results) else 1)
"""
    with tempfile.TemporaryDirectory(prefix="conceptlens-generation-") as temp_dir:
        test_file = Path(temp_dir) / "evaluate_generated.py"
        test_file.write_text(f"{generated_code}\n\n{test_source}\n\n{runner}\n", encoding="utf-8")
        completed = _run_evaluation_file(test_file, temp_dir)
    marker = "__CONCEPTLENS_TEST_RESULTS__="
    encoded_results = next((line[len(marker):] for line in completed.stdout.splitlines() if line.startswith(marker)), "[]")
    try:
        test_results = json.loads(encoded_results)
    except json.JSONDecodeError:
        test_results = []
    return {
        "testsPassed": sum(1 for item in test_results if item.get("passed")),
        "testsTotal": len(test_results),
        "testResults": test_results,
        "fullSuccess": completed.returncode == 0 and bool(test_results),
        "stdout": completed.stdout[-2000:],
        "stderr": completed.stderr[-2000:],
    }


def evaluate_generation(payload: dict[str, Any]) -> dict[str, Any]:
    generation_id = str(payload.get("generationId") or "")
    record = _read_record(GENERATION_RECORDS_DIR, generation_id)
    case_id = str(record.get("caseId"))
    evaluation = HIDDEN_EVALUATIONS.get(case_id)
    if not evaluation:
        raise KeyError("No hidden tests are configured for this generation case.")
    ground_truth = build_candidate_payload(case_id, f"code_{int(SINGLE_REFERENCE_CASE_CONFIG[case_id]['hiddenGroundTruthCodeIdx'])}")
    api_metrics = _api_overlap_metrics(str(record["generatedCode"]), str(ground_truth["rawCode"]))
    if not GENERATION_EXECUTION_ENABLED:
        result = {"status": "evaluation_unavailable", "message": "Hidden-test execution is disabled. Set GENERATION_EXECUTION_ENABLED=1 only in a trusted local evaluation environment.", "generationId": generation_id, **api_metrics}
        evaluation_record = {"id": f"evaluation_{uuid.uuid4().hex}", "kind": "evaluation", "createdAt": time.time(), "caseId": record["caseId"], "condition": record["condition"], "selectionId": record.get("selectionId"), "sessionId": record.get("sessionId", ""), "participantId": record.get("participantId", ""), "caseAttemptId": record.get("caseAttemptId", ""), **result}
        _write_record(GENERATION_RECORDS_DIR, evaluation_record)
        return result
    try:
        test_result = _run_hidden_tests(str(record["generatedCode"]), evaluation)
    except ValueError as exc:
        test_result = {
            "testsPassed": 0,
            "testsTotal": 1,
            "testResults": [{"name": "generated_function_validation", "passed": False, "error": str(exc)}],
            "fullSuccess": False,
            "stdout": "",
            "stderr": "",
        }
    result = {
        "status": "ok",
        "generationId": generation_id,
        "testsPassed": test_result["testsPassed"],
        "testsTotal": test_result["testsTotal"],
        "passRate": round(test_result["testsPassed"] / test_result["testsTotal"], 4) if test_result["testsTotal"] else 0.0,
        "fullSuccess": test_result["fullSuccess"],
        "testResults": test_result["testResults"],
        "stdout": test_result["stdout"],
        "stderr": test_result["stderr"],
        **api_metrics,
    }
    evaluation = {"id": f"evaluation_{uuid.uuid4().hex}", "kind": "evaluation", "createdAt": time.time(), "caseId": record["caseId"], "condition": record["condition"], "selectionId": record.get("selectionId"), "sessionId": record.get("sessionId", ""), "participantId": record.get("participantId", ""), "caseAttemptId": record.get("caseAttemptId", ""), **result}
    _write_record(GENERATION_RECORDS_DIR, evaluation)
    return result


def run_batch_evaluation(case_ids: list[str], conditions: list[str], interactive_selections: dict[str, str] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case_id in case_ids:
        for condition in conditions:
            selection_id = (interactive_selections or {}).get(case_id) if condition == "interactive_rag" else None
            try:
                generated = generate_code({"caseId": case_id, "condition": condition, "selectionId": selection_id})
                evaluated = evaluate_generation({"generationId": generated["generationId"]})
                results.append({"case_id": case_id, "condition": condition, "candidate_id": generated.get("contextCandidateId"), "generation_id": generated["generationId"], "tests_passed": evaluated.get("testsPassed"), "tests_total": evaluated.get("testsTotal"), "pass_rate": evaluated.get("passRate"), "full_success": evaluated.get("fullSuccess"), "status": evaluated["status"]})
            except Exception as exc:
                results.append({"case_id": case_id, "condition": condition, "status": "error", "error": str(exc)})
    return results


def export_batch_results(results: list[dict[str, Any]]) -> tuple[Path, Path]:
    GENERATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = GENERATION_RESULTS_DIR / "generation_evaluation.json"
    csv_path = GENERATION_RESULTS_DIR / "generation_evaluation.csv"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["case_id", "condition", "candidate_id", "generation_id", "tests_passed", "tests_total", "pass_rate", "full_success", "status", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path
