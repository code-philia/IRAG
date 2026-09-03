#!/usr/bin/env python3
"""Export the current study-dev reference candidates as readable Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


CASES = (
    ("csn_8884", "Try AST dead-code cleanup"),
    ("csn_3846", "Decorator redefinition"),
    ("csn_42", "Cloud SQL delete completion"),
    ("csn_9388", "URL query removal"),
)

TEST_FILES = (
    ("csn_8884", Path("backend/app/generation_tests/csn_8884.py")),
    ("csn_3846", Path("backend/app/generation_tests/csn_3846.py")),
    ("csn_42", Path("backend/app/generation_tests/csn_42.py")),
    ("csn_9388", Path("backend/app/generation_tests/csn_9388.py")),
)


def get_json(base_url: str, path: str, payload: dict | None = None) -> dict:
    url = f"{base_url.rstrip('/')}{path}"
    if payload is None:
        request = Request(url)
    else:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def markdown_for_case(base_url: str, case_id: str, title: str) -> str:
    session = get_json(base_url, "/api/session/load", {"testId": case_id, "topK": 20})
    candidates = sorted(session.get("candidates", []), key=lambda item: int(item.get("rank", 0)))
    sections = [f"## {case_id} - {title}\n", f"Candidates exported: {len(candidates)}\n"]
    for candidate in candidates:
        candidate_id = str(candidate["id"])
        detail = get_json(base_url, f"/api/candidates/{candidate_id}?test_id={case_id}")
        metadata = detail.get("metadata") or candidate.get("metadata") or {}
        rank = int(candidate.get("rank", detail.get("rank", 0)))
        name = metadata.get("funcName") or candidate_id
        repo = metadata.get("repo") or "unknown repository"
        path = metadata.get("path") or "unknown path"
        url = metadata.get("url") or ""
        code = detail.get("rawCode") or ""
        sections.extend(
            [
                f"### Rank {rank} - `{candidate_id}` - `{name}`\n",
                f"- Repository: `{repo}`\n",
                f"- Path: `{path}`\n",
                f"- Similarity: `{float(candidate.get('similarity', detail.get('similarity', 0.0))):.6f}`\n",
                f"- Source URL: {url}\n",
                "\n```python\n",
                code.rstrip(),
                "\n```\n",
            ]
        )
    return "\n".join(sections)


def markdown_for_tests() -> str:
    sections = [
        "## Generation Tests\n",
        "The following behavior-oriented tests are the current generation evaluation cases. `csn_11772` is intentionally omitted.\n",
    ]
    for case_id, path in TEST_FILES:
        sections.extend(
            [
                f"### {case_id} tests\n",
                f"Source: `{path}`\n",
                "\n```python\n",
                path.read_text(encoding="utf-8").rstrip(),
                "\n```\n",
            ]
        )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://1.94.111.205")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/csn_reference_candidates.md"),
    )
    args = parser.parse_args()

    sections = [
        "# Study Dev Reference Candidates\n",
        "> Snapshot of the 20 participant-facing candidates for the four active cases.\n",
        "> Generated from the study-dev candidate API; `csn_11772` is intentionally omitted.\n",
    ]
    for case_id, title in CASES:
        sections.append(markdown_for_case(args.base_url, case_id, title))
    sections.append(markdown_for_tests())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
