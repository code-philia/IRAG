from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
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
)
from .data_service import (
    SINGLE_REFERENCE_CASE_CONFIG,
    build_candidate_payload,
    build_session_payload,
    is_hidden_reference_candidate,
    is_single_reference_case,
)


PROMPT_VERSION = "single-reference-interactive-rag-v1"
SYSTEM_PROMPT = "You are a careful Python programmer. Return only complete Python code, without Markdown fences or explanation."

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
}


def _public_task(case_id: str) -> dict[str, Any]:
    if not is_single_reference_case(case_id):
        raise KeyError(f"No single-reference generation case is configured for {case_id}.")
    session = build_session_payload(str(case_id), top_k=20)
    return {
        "caseId": str(case_id),
        "query": session["query"]["rawText"],
        "language": "python",
    }


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
    session = build_session_payload(case_id, top_k=20)
    visible_ids = {str(item["id"]) for item in session.get("candidates", [])}
    if selected_candidate_id not in visible_ids or is_hidden_reference_candidate(case_id, selected_candidate_id):
        raise ValueError("The selected candidate is not an available reference for this case.")
    candidate = build_candidate_payload(case_id, selected_candidate_id)
    record = {
        "id": f"selection_{uuid.uuid4().hex}",
        "kind": "reference_selection",
        "createdAt": time.time(),
        "caseId": case_id,
        "selectedCandidateId": candidate["id"],
        "selectedRank": payload.get("selectedRank"),
        "selectedScore": payload.get("selectedScore", candidate.get("similarity")),
        "interactionUsed": bool(payload.get("interactionUsed", False)),
        "model": str(payload.get("model") or "xsearch"),
        "protocol": "single_reference",
        "targetReferenceMatched": candidate["codeIdx"] == SINGLE_REFERENCE_CASE_CONFIG[case_id]["targetReferenceCodeIdx"],
        "candidate": {
            "id": candidate["id"],
            "codeIdx": candidate["codeIdx"],
            "rawCode": candidate["rawCode"],
            "metadata": candidate.get("metadata", {}),
        },
    }
    selection_id = _write_record(GENERATION_RECORDS_DIR, record)
    public_selection = {key: value for key, value in record.items() if key != "targetReferenceMatched"}
    return {"status": "ok", "selectionId": selection_id, "task": task, "selection": public_selection}


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
        if selection.get("caseId") != case_id:
            raise ValueError("The selected retrieval record belongs to a different case.")
        return dict(selection["candidate"])
    raise ValueError("condition must be no_rag, automatic_rag, or interactive_rag.")


def _generation_prompt(task: dict[str, Any], context: dict[str, Any] | None) -> str:
    prompt = f"Programming Task:\n{task['query']}\n"
    if context:
        prompt += f"\nRetrieved reference code:\n{context['rawCode']}\n"
    prompt += "\nImplement the requested functionality. Use the retrieved example only when it is relevant."
    return prompt


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
    if GENERATION_PROVIDER != "openai_compatible" or not (GENERATION_API_URL and GENERATION_API_KEY and GENERATION_MODEL):
        raise RuntimeError("generator_unavailable: configure GENERATION_API_URL, GENERATION_API_KEY, and GENERATION_MODEL for the fixed OpenAI-compatible generator.")
    request_payload = {
        "model": GENERATION_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "temperature": GENERATION_TEMPERATURE,
        "max_tokens": GENERATION_MAX_TOKENS,
    }
    request = Request(
        GENERATION_API_URL,
        data=json.dumps(request_payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {GENERATION_API_KEY}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return _extract_generated_code(json.loads(response.read().decode("utf-8")))
    except HTTPError as exc:
        raise RuntimeError(f"generator_unavailable: provider returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"generator_unavailable: provider request failed: {exc.reason}") from exc


def generate_code(payload: dict[str, Any]) -> dict[str, Any]:
    case_id = str(payload.get("caseId") or "")
    task = _public_task(case_id)
    condition = str(payload.get("condition") or "interactive_rag")
    context = _resolve_context(case_id, condition, payload.get("selectionId"))
    prompt = _generation_prompt(task, context)
    started = time.perf_counter()
    generated_code = _call_generator(prompt)
    record = {
        "id": f"generation_{uuid.uuid4().hex}",
        "kind": "generation",
        "createdAt": time.time(),
        "caseId": case_id,
        "condition": condition,
        "selectionId": payload.get("selectionId"),
        "contextCandidateId": context.get("id") if context else None,
        "contextCandidate": context,
        "model": GENERATION_MODEL,
        "provider": GENERATION_PROVIDER,
        "temperature": GENERATION_TEMPERATURE,
        "maxTokens": GENERATION_MAX_TOKENS,
        "promptVersion": PROMPT_VERSION,
        "prompt": prompt,
        "generatedCode": generated_code,
        "generationTime": round(time.perf_counter() - started, 4),
    }
    generation_id = _write_record(GENERATION_RECORDS_DIR, record)
    return {**record, "generationId": generation_id, "status": "ok"}


def evaluate_generation(payload: dict[str, Any]) -> dict[str, Any]:
    generation_id = str(payload.get("generationId") or "")
    record = _read_record(GENERATION_RECORDS_DIR, generation_id)
    if not GENERATION_EXECUTION_ENABLED:
        return {"status": "evaluation_unavailable", "message": "Hidden-test execution is disabled. Set GENERATION_EXECUTION_ENABLED=1 only in a trusted local evaluation environment.", "generationId": generation_id}
    evaluation = HIDDEN_EVALUATIONS.get(str(record.get("caseId")))
    if not evaluation:
        raise KeyError("No hidden tests are configured for this generation case.")
    with tempfile.TemporaryDirectory(prefix="conceptlens-generation-") as temp_dir:
        test_file = Path(temp_dir) / "evaluate_generated.py"
        test_file.write_text(f"{record['generatedCode']}\n\n{evaluation['tests']}\n", encoding="utf-8")
        completed = subprocess.run([sys.executable, "-I", str(test_file)], cwd=temp_dir, capture_output=True, text=True, timeout=15, check=False)
    result = {"status": "ok", "generationId": generation_id, "testsPassed": 1 if completed.returncode == 0 else 0, "testsTotal": 1, "passRate": 1.0 if completed.returncode == 0 else 0.0, "fullSuccess": completed.returncode == 0, "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-2000:]}
    evaluation = {"id": f"evaluation_{uuid.uuid4().hex}", "kind": "evaluation", "createdAt": time.time(), "caseId": record["caseId"], "condition": record["condition"], **result}
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
