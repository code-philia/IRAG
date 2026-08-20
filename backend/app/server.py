from __future__ import annotations

import json
import threading
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import DEFAULT_DATASET_PATH, DEFAULT_EXPERIMENT_ID, DEFAULT_MATCH_PATH, LATEST_STEP_CHECKPOINT_PATH
from .data_service import build_candidate_payload, build_session_payload, get_available_tests
from .dynavis_service import build_dynavis_graph
from .gradient_attribution_service import build_gradient_attribution
from .intervention_service import (
    apply_adapter_to_candidate_payload,
    apply_adapter_to_session_payload,
    apply_drag_rerank,
    apply_manual_link,
    reset_interventions,
)
from .log_service import append_event
from .training_attribution_service import build_token_pair_attribution


BOOTSTRAP_PREWARM_TEST_IDS = ("csn_11078", "csn_11087")


@lru_cache(maxsize=16)
def _cached_initial_bootstrap(test_id: str, top_k: int) -> dict:
    session = apply_adapter_to_session_payload(build_session_payload(test_id, top_k))
    first = next(iter(session.get("candidates") or []), None)
    if not first:
        raise ValueError("No candidates available for this test.")
    candidate_id = str(first["id"])
    candidate = apply_adapter_to_candidate_payload(build_candidate_payload(test_id, candidate_id))
    graph = build_dynavis_graph(test_id, candidate_id, candidate=candidate)
    return {"session": session, "candidate": candidate, "graph": graph}


def _prewarm_bootstrap_cache() -> None:
    for test_id in BOOTSTRAP_PREWARM_TEST_IDS:
        try:
            _cached_initial_bootstrap(test_id, 20)
        except Exception as exc:
            print(f"Bootstrap prewarm skipped for {test_id}: {exc}")


class ApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({"status": "ok"})

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            path = parsed.path

            if path == "/api/health":
                self._send_json({"status": "ok"})
            elif path == "/api/experiments":
                self._send_json(
                    {
                        "experiments": [
                            {
                                "id": DEFAULT_EXPERIMENT_ID,
                                "name": "XSearch User Study Python",
                                "datasetPath": str(DEFAULT_DATASET_PATH),
                                "matchPath": str(DEFAULT_MATCH_PATH),
                                "checkpointPath": str(LATEST_STEP_CHECKPOINT_PATH),
                                "hasModel": LATEST_STEP_CHECKPOINT_PATH.exists(),
                                "modelEnabled": True,
                                "projectionMethod": "DynaVis",
                            }
                        ],
                        "tests": get_available_tests(),
                    }
                )
            elif path.startswith("/api/candidates/"):
                candidate_id = path.rsplit("/", 1)[-1]
                test_id = query.get("test_id", [""])[0]
                self._send_json(apply_adapter_to_candidate_payload(build_candidate_payload(test_id, candidate_id)))
            elif path == "/api/visualize/graph":
                test_id = query.get("test_id", [""])[0]
                candidate_id = query.get("candidate_id", [""])[0]
                epoch = int(query.get("epoch", ["4"])[0])
                self._send_json(build_dynavis_graph(test_id, candidate_id, epoch))
            else:
                self._send_json({"error": f"Unknown endpoint: {path}"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")

            if parsed.path == "/api/session/load":
                test_id = str(data.get("testId") or "")
                top_k = int(data.get("topK") or 10)
                self._send_json(apply_adapter_to_session_payload(build_session_payload(test_id, top_k)))
            elif parsed.path == "/api/session/bootstrap":
                test_id = str(data.get("testId") or "")
                top_k = int(data.get("topK") or 20)
                reset_interventions({"testId": test_id})
                self._send_json(_cached_initial_bootstrap(test_id, top_k))
            elif parsed.path == "/api/logs/events":
                self._send_json(append_event(data))
            elif parsed.path == "/api/intervention/manual-link":
                self._send_json(apply_manual_link(data))
            elif parsed.path == "/api/intervention/drag-rerank":
                self._send_json(apply_drag_rerank(data))
            elif parsed.path == "/api/intervention/reset":
                self._send_json(reset_interventions(data))
            elif parsed.path == "/api/diagnostics/token-pair-attribution":
                self._send_json(build_token_pair_attribution(data))
            elif parsed.path == "/api/diagnostics/token-pair-gradient-attribution":
                self._send_json(build_gradient_attribution(data))
            elif parsed.path == "/api/intervention/re-rank":
                self._send_json(
                    {
                        "status": "not_implemented",
                        "message": "Interactive re-ranking is reserved for the next model/cache integration stage.",
                    },
                    HTTPStatus.NOT_IMPLEMENTED,
                )
            else:
                self._send_json({"error": f"Unknown endpoint: {parsed.path}"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def run(host: str = "127.0.0.1", port: int = 8765):
    server = ThreadingHTTPServer((host, port), ApiHandler)
    threading.Thread(target=_prewarm_bootstrap_cache, name="bootstrap-prewarm", daemon=True).start()
    print(f"XSearch CHI backend listening on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
