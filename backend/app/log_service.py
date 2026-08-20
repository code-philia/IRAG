from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .config import LOG_DIR


def append_event(event: dict[str, Any]) -> dict[str, str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    session_id = str(event.get("sessionId") or "anonymous")
    payload = dict(event)
    payload["timestamp"] = datetime.now().isoformat()
    with open(LOG_DIR / f"{session_id}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return {"status": "ok"}
