"""Loopback HTTP control for caller-defined voice events."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading


class _ControlHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], callbacks: Mapping[str, Callable[[], None]]) -> None:
        super().__init__(server_address, _ControlHandler)
        self.event_callbacks = dict(callbacks)


class _ControlHandler(BaseHTTPRequestHandler):
    server: _ControlHttpServer

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
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
    def __init__(self, host: str, port: int, callbacks: Mapping[str, Callable[[], None]]) -> None:
        self._server = _ControlHttpServer((host, port), callbacks)
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
