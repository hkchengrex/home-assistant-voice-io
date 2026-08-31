from http.client import HTTPConnection
from io import BytesIO
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import av
import pytest

from ha_voice.audio import Audio, encode_wav
from ha_voice.config import load_config
from ha_voice.studio_server import StudioServer


@pytest.fixture
def server(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "commands.toml")
    instance = StudioServer(("127.0.0.1", 0), config, tmp_path / "recordings")
    worker = threading.Thread(target=instance.serve_forever, daemon=True)
    worker.start()
    yield instance
    instance.shutdown()
    instance.server_close()
    worker.join(timeout=5)


def request(server, action, payload=None, *, headers=None, method=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload) if payload is not None else None
    defaults = {"X-Voice-IO": "response-studio", "Content-Type": "application/json"}
    defaults.update(headers or {})
    connection = HTTPConnection(*server.server_address, timeout=5)
    try:
        connection.request(method or ("GET" if body is None else "POST"), action, body, defaults)
        response = connection.getresponse()
        raw = response.read()
        content = json.loads(raw) if response.getheader("Content-Type", "").startswith("application/json") else raw
        return response.status, content, dict(response.getheaders())
    finally:
        connection.close()


@pytest.mark.parametrize("path", ["/responses", "/responses.js", "/responses.css"])
def test_static_response_page(server, path):
    status, body, headers = request(server, path)
    assert status == 200 and body
    assert "script-src 'self'" in headers["Content-Security-Policy"]


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example"}, {"Origin": "https://evil.example"},
    {"Host": "[invalid"}, {"Origin": "http://[invalid"}, {"Origin": "null"},
])
def test_cross_origin_requests_cannot_read_private_workspace(server, headers):
    status, _, _ = request(server, "/api/responses/state", headers=headers)
    assert status == 403
    assert server._responses is None


def test_writes_require_explicit_header(server):
    status, _, _ = request(server, "/api/responses/phrases", {"group": "start", "text": "Hi."},
                           headers={"X-Voice-IO": ""})
    assert status == 403
    assert server._responses is None


@pytest.mark.parametrize("payload,headers", [([], {}), (b"{invalid", {}),
    ({}, {"Content-Type": "text/plain"}), (b"{}", {"Content-Length": "70000"})])
def test_invalid_payloads_are_rejected(server, payload, headers):
    assert request(server, "/api/responses/phrases", payload, headers=headers)[0] == 400


def test_http_generate_review_publish_and_audio(server, tmp_path):
    status, state, _ = request(server, "/api/responses/state")
    assert status == 200 and len(state["phrases"]) == 32
    assert not state["reference"]
    signal = (.1 * np.sin(np.arange(48000) * 2 * np.pi * 180 / 24000)).astype(np.float32)
    audio = encode_wav(Audio(signal, 24000))
    assert request(server, "/api/responses/reference", audio)[0] == 200
    assert request(server, "/api/responses/reference")[1] == audio
    source = tmp_path / "synthetic.wav"
    source.write_bytes(audio)
    studio = server.response_workspace()
    studio.cloner_factory = lambda: SimpleNamespace(generate_batch=lambda **kw:
        SimpleNamespace(audio_paths=[source] * len(kw["texts"]), remote_metrics={}))
    payload = {"request_id": uuid4().hex, "phrase_ids": [state["phrases"][0]["id"]],
               "takes": 2, "consent_to_upload": True, "settings": {"speed": .9, "guidance_scale": 1.5}}
    status, result, _ = request(server, "/api/responses/generate", payload)
    assert status == 202 and result["job"]["total"] == 2
    studio.thread.join(timeout=5)
    assert not studio.thread.is_alive()
    candidates = request(server, "/api/responses/state")[1]["candidates"]
    assert len(candidates) == 2
    chosen = candidates[0]["id"]
    status, body, headers = request(server, candidates[0]["audio_url"])
    assert status == 200 and body.startswith(b"RIFF")
    assert headers["Content-Type"] == "audio/wav"
    assert request(server, "/api/responses/review", {"candidate_ids": [chosen], "status": "kept"})[0] == 200
    status, published, _ = request(server, "/api/responses/publish", {"candidate_ids": [chosen]})
    assert status == 200 and not published["listener_updated"]
    assert len(list(server.assets_dir.glob("*.wav"))) == 1
    assert request(server, "/api/responses/unpublish", {"id": chosen})[0] == 200
    assert not list(server.assets_dir.glob("*.wav"))
    assert studio.audio_path(chosen).exists()
    assert request(server, "/api/responses/audio/../secret")[0] == 400
    assert request(server, "/api/responses/missing")[0] == 404


@pytest.mark.parametrize("container_format,codec", [("wav", "pcm_f32le"), ("mp3", "libmp3lame"), ("mp4", "aac")])
def test_upload_converts_before_preview_and_failed_upload_preserves_reference(server, container_format, codec):
    output = BytesIO()
    with av.open(output, "w", format=container_format) as container:
        stream = container.add_stream(codec, rate=48000)
        stream.layout = "stereo"
        frame = av.AudioFrame.from_ndarray(np.full((2, 96000), .1, dtype=np.float32), format="fltp", layout="stereo")
        frame.sample_rate = 48000
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)
    status, result, _ = request(server, "/api/responses/reference", output.getvalue(),
                                headers={"Content-Type": "application/octet-stream"})
    assert status == 200
    reference = result["state"]["reference"]
    assert reference["sample_rate"] == 24000
    preview = request(server, "/api/responses/reference")[1]
    import wave
    with wave.open(BytesIO(preview), "rb") as audio:
        assert audio.getsampwidth() == 2 and audio.getnchannels() == 1
        assert audio.getframerate() == 24000
        assert audio.getnframes() / 24000 == pytest.approx(2, abs=.08)
    assert request(server, "/api/responses/reference", b"broken audio")[0] == 400
    assert request(server, "/api/responses/state")[1]["reference"] == reference
    assert request(server, "/api/responses/reference")[1] == preview
