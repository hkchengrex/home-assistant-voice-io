"""Loopback HTTP control for caller-defined voice events."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
import json


class _ControlHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], callbacks: Mapping[str, Callable[[], None]], status_callback=None) -> None:
        super().__init__(server_address, _ControlHandler)
        self.event_callbacks = dict(callbacks)
        self.status_callback = status_callback


class _ControlHandler(BaseHTTPRequestHandler):
    server: _ControlHttpServer

    def _respond(self, status: int, body: bytes, content_type="text/plain; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/voice/status" and self.server.status_callback is not None:
            self._respond(200, json.dumps(self.server.status_callback()).encode(), "application/json")
            return
        self._respond(200, b"ok\n") if self.path == "/health" else self._respond(404, b"not found\n")

    def do_POST(self) -> None:  # noqa: N802
        callback = self.server.event_callbacks.get(self.path)
        if callback is None:
            self._respond(404, b"not found\n")
            return
        try:
            callback()
        except RuntimeError as exc:
            self._respond(503, f"{exc}\n".encode())
            return
        self._respond(202, b"queued\n")

    def log_message(self, format: str, *args: object) -> None:
        del format, args


class VoiceControlServer:
    def __init__(self, host: str, port: int, callbacks: Mapping[str, Callable[[], None]], status_callback=None) -> None:
        self._server = _ControlHttpServer((host, port), callbacks, status_callback)
        self._thread = threading.Thread(target=self._server.serve_forever, name="voice-control-server", daemon=True)

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
