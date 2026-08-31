from http.client import HTTPConnection
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ha_voice.config import load_config
from ha_voice.diagnostics import TriggerCaptureQueue
from ha_voice.studio_server import StudioServer


@pytest.fixture
def server(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "commands.toml")
    instance = StudioServer(("127.0.0.1", 0), config, tmp_path / "recordings")
    instance.update_calls = []
    instance.listener_manager = SimpleNamespace(update=lambda **kw: (
        instance.update_calls.append(kw) or SimpleNamespace(updated=True, error=None, waiting_for_templates=False)))
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    worker.join(timeout=5)


def request(server, path, body=None, headers=None):
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request("GET" if body is None else "POST", path, json.dumps(body) if body is not None else None,
                           {"Content-Type": "application/json", "X-Voice-IO": "diagnostic-review", **(headers or {})})
        response = connection.getresponse()
        raw = response.read()
        return response.status, json.loads(raw) if response.getheader("Content-Type", "").startswith("application/json") else raw
    finally:
        connection.close()


def capture(server):
    samples = (.2 * np.sin(np.arange(16000) * 2 * np.pi * 180 / 16000)).astype(np.float32)
    other_process_queue = TriggerCaptureQueue(server.recordings_dir, sample_rate=16000)
    return other_process_queue.capture(samples, {"kind": "ignored", "accepted": False})


def test_http_enable_capture_replay_teach_and_disable(server):
    assert request(server, "/api/diagnostics")[1]["review_mode"]["active"] is False
    assert request(server, "/api/diagnostics/review-mode", {"enabled": True})[0] == 200
    event = capture(server)
    state = request(server, "/api/diagnostics")[1]
    assert state["count"] == 1
    assert request(server, state["events"][0]["attempt"]["audio_url"])[1].startswith(b"RIFF")
    status, result = request(server, f"/api/diagnostics/{event}/teach", {"clip": "attempt", "label": "lights_on"})
    assert status == 200 and result["listener_updated"]
    assert len(server.update_calls) == 1
    assert (server.recordings_dir / "lights_on" / result["filename"]).exists()
    assert request(server, "/api/diagnostics")[1]["events"][0]["attempt"]["taught_as"] == "lights_on"
    assert request(server, "/api/diagnostics/review-mode", {"enabled": False})[0] == 200
    assert capture(server) is None


@pytest.mark.parametrize("headers", [{"X-Voice-IO": ""}, {"Origin": "http://evil.example"}, {"Host": "evil.example"}, {"Host": "[bad"}])
def test_review_requires_same_origin_and_explicit_header(server, headers):
    assert request(server, "/api/diagnostics/review-mode", {"enabled": True}, headers)[0] == 403
    assert request(server, "/api/diagnostics/invalid/teach", {"clip": "start", "label": "lights_on"}, headers)[0] == 403
    assert not server.trigger_captures.review_status()["active"]


@pytest.mark.parametrize("payload", [[], {}, {"enabled": "true"}, {"enabled": 1}])
def test_invalid_review_requests_do_not_enable_capture(server, payload):
    assert request(server, "/api/diagnostics/review-mode", payload)[0] == 400
    assert not server.trigger_captures.review_status()["active"]


def test_invalid_teaching_does_not_reload_listener(server):
    assert request(server, "/api/diagnostics/bad/teach", {"clip": "attempt", "label": "lights_on"})[0] == 400
    assert not server.update_calls


def test_http_polling_distinguishes_expired_and_rejected_commands(server, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("ha_voice.diagnostics.time.time", lambda: clock[0])
    queue = TriggerCaptureQueue(server.recordings_dir, sample_rate=16000, config=server.app_config)
    samples = np.zeros(16000, dtype=np.float32)
    queue.capture(samples, {"kind": "start_phrase", "accepted": True})
    assert request(server, "/api/diagnostics")[1]["events"][0]["command_status"] == "waiting"
    clock[0] += server.app_config.start_phrase.command_timeout_seconds
    assert request(server, "/api/diagnostics")[1]["events"][0]["command_status"] == "nothing_detected"
    queue.capture(samples, {"kind": "command", "accepted": False, "score": 3, "margin": 0})
    event = request(server, "/api/diagnostics")[1]["events"][0]
    assert event["command_status"] == "rejected"
    assert event["command"]["rejection_reason"] == "margin"
    assert not server.update_calls


def test_teaching_requires_stopped_studio_test(server, monkeypatch):
    monkeypatch.setattr(server.listener, "snapshot", lambda: {"running": True})
    assert request(server, "/api/diagnostics/bad/teach", {"clip": "attempt", "label": "lights_on"})[0] == 409
    assert not server.update_calls
