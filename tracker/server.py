"""Lightweight local HTTP server for two-way project progress tracking."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TRACKER_DIR = Path(__file__).resolve().parent
INDEX_PATH = TRACKER_DIR / "index.html"
PROGRESS_PATH = TRACKER_DIR / "progress.json"

_FILE_LOCK = threading.Lock()


def _read_progress() -> dict[str, Any]:
    with _FILE_LOCK:
        with PROGRESS_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh)


def _write_progress_atomic(data: dict[str, Any]) -> None:
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=".progress_", suffix=".tmp", dir=TRACKER_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp_name, PROGRESS_PATH)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _update_node(
    node: dict[str, Any], target_id: str, done: bool, checked_by: str
) -> bool:
    if node.get("id") == target_id and node.get("type") == "task":
        node["done"] = done
        node["checked_by"] = checked_by if done else None
        node["updated_at"] = (
            datetime.now().astimezone().isoformat(timespec="seconds") if done else None
        )
        return True

    for child in node.get("children", []):
        if _update_node(child, target_id, done, checked_by):
            return True
    return False


class TrackerHandler(BaseHTTPRequestHandler):
    """Serve the OSINT-style tree UI and JSON toggle API."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return  # Keep console output quiet for clean background operation

    def _send_bytes(
        self, status: HTTPStatus, body: bytes, content_type: str
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, raw, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            if not INDEX_PATH.exists():
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "index.html not found"})
                return
            self._send_bytes(
                HTTPStatus.OK,
                INDEX_PATH.read_bytes(),
                "text/html; charset=utf-8",
            )
            return

        if path == "/api/progress":
            try:
                data = _read_progress()
            except FileNotFoundError:
                self._send_json(
                    HTTPStatus.NOT_FOUND, {"error": "progress.json not found"}
                )
                return
            self._send_json(HTTPStatus.OK, data)
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/api/toggle":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown route: {path}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length)
            req = json.loads(raw_body.decode("utf-8"))
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"})
            return

        target_id = req.get("id")
        done = req.get("done")
        checked_by = req.get("checked_by") or "user"
        if not isinstance(target_id, str) or not isinstance(done, bool):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "Payload must include string 'id' and boolean 'done'"},
            )
            return
        if checked_by not in ("user", "ai"):
            checked_by = "user"

        with _FILE_LOCK:
            with PROGRESS_PATH.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not _update_node(data, target_id, done, checked_by):
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": f"Task id '{target_id}' not found"},
                )
                return
            _write_progress_atomic(data)

        self._send_json(HTTPStatus.OK, data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local progress tracker server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TrackerHandler)
    print(f"Progress tracker listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
