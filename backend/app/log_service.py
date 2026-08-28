from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any

from .config import LOG_DIR, STUDY_SESSION_DIR


LOG_LOCK = threading.RLock()
SINGLE_SUBMISSION_EVENTS = {"confidence_submit", "difficulty_submit", "task_end"}


def append_event(event: dict[str, Any]) -> dict[str, str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_id = str(event.get("sessionId") or "anonymous")
    payload = {
        "eventId": str(event.get("eventId") or f"event_{datetime.now().timestamp():.6f}"),
        "sessionId": session_id,
        "participantId": str(event.get("participantId") or ""),
        "condition": str(event.get("condition") or event.get("appMode") or "demo"),
        "caseId": str(event.get("caseId") or event.get("testId") or ""),
        "eventType": str(event.get("eventType") or "unknown"),
        "timestamp": datetime.now().isoformat(),
        "payload": dict(event.get("eventData") or {}),
    }
    log_path = LOG_DIR / f"{session_id}.jsonl"
    with LOG_LOCK:
        if payload["eventType"] in SINGLE_SUBMISSION_EVENTS and payload["payload"].get("caseAttemptId") and log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                existing = json.loads(line)
                if (
                    existing.get("eventType") == payload["eventType"]
                    and str((existing.get("payload") or {}).get("caseAttemptId")) == str(payload["payload"]["caseAttemptId"])
                ):
                    return {"status": "duplicate"}
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"status": "ok"}


def store_study_session(session: dict[str, Any]) -> dict[str, str]:
    STUDY_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_LOCK:
        (STUDY_SESSION_DIR / f"{session['sessionId']}.json").write_text(
            json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return {"status": "ok", "sessionId": str(session["sessionId"])}
