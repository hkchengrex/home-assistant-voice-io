import json
import threading
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest

from ha_voice.actions import ActionResult
from ha_voice.continuous import ContinuousListener
from ha_voice.control_server import VoiceControlServer
from ha_voice.voice_mode import VoiceMode


def setup_mode(monkeypatch, tmp_path):
    calls = []
    listener = ContinuousListener()
    listener._state["running"] = True
    responses = SimpleNamespace(play=lambda group, **kwargs: calls.append(group))
    monkeypatch.setattr("ha_voice.voice_mode.play_beeps", lambda count, **kwargs: calls.append(count))
    mode = VoiceMode(listener, responses, state_file=tmp_path / "mode.json")
    return mode, listener, calls


def drain(listener):
    item = listener._take_external_feedback()
    assert item is not None
    item.callback()
    listener._active_feedback = None


def test_toggle_beeps_paused_commands_and_persistent_state(monkeypatch, tmp_path):
    mode, listener, calls = setup_mode(monkeypatch, tmp_path)
    mode.set_enabled()
    drain(listener)
    mode.queue_response("ack")
    drain(listener)
    assert calls == [2, 1]
    assert not mode.status()["enabled"]
    assert not listener.input_is_current(listener.input_epoch)
    restored = VoiceMode(ContinuousListener(), None, state_file=mode.state_file)
    assert not restored.status()["enabled"]
    mode.set_enabled()
    drain(listener)
    mode.queue_response("ack")
    drain(listener)
    assert calls == [2, 1, 1, "ack"]
    assert json.loads(mode.state_file.read_text()) == {"enabled": True}


def test_pause_cancels_active_speech_and_stale_pending(monkeypatch, tmp_path):
    mode, listener, calls = setup_mode(monkeypatch, tmp_path)
    started, ended = threading.Event(), threading.Event()
    def play(group, *, cancelled):
        started.set()
        assert cancelled.wait(2)
        ended.set()
    mode.responses.play = play
    mode.queue_response("active")
    active = listener._take_external_feedback()
    thread = threading.Thread(target=active.callback)
    thread.start()
    assert started.wait(1)
    mode.queue_response("stale")
    mode.set_enabled(False)
    thread.join(2)
    assert ended.is_set()
    drain(listener)
    assert calls == [2]
    assert listener._take_external_feedback() is None


def test_inflight_recognition_cannot_execute_after_pause_resume(monkeypatch, tmp_path):
    mode, listener, calls = setup_mode(monkeypatch, tmp_path)
    resets = []
    mode.reset_recognition = lambda: resets.append(True)
    entered, finish = threading.Event(), threading.Event()
    def recognize(samples):
        entered.set()
        assert finish.wait(2)
        result = mode.execute(lambda name: calls.append(name), "command")
        assert result.suppress_feedback
        assert mode.play_spoken("ack") is False
    thread = threading.Thread(target=lambda: mode.recognize(recognize, None))
    thread.start()
    assert entered.wait(1)
    mode.set_enabled(False)
    mode.set_enabled(True)
    finish.set()
    thread.join(2)
    assert not thread.is_alive()
    assert calls == []
    mode.recognize(lambda samples: mode.execute(lambda name: calls.append(name), "fresh"), None)
    assert calls == ["fresh"]
    assert len(resets) == 2


def test_pause_interrupts_recognized_spoken_feedback(monkeypatch, tmp_path):
    mode, listener, calls = setup_mode(monkeypatch, tmp_path)
    started = threading.Event()
    def play(group, *, cancelled):
        started.set()
        assert cancelled.wait(2)
    mode.responses.play = play
    def recognize(samples):
        assert mode.play_spoken("ack") is False
    thread = threading.Thread(target=lambda: mode.recognize(recognize, None))
    thread.start()
    assert started.wait(1)
    mode.set_enabled(False)
    thread.join(2)
    assert not thread.is_alive()
    drain(listener)
    assert calls == [2]


def test_failed_state_write_does_not_change_mode(monkeypatch, tmp_path):
    mode, listener, calls = setup_mode(monkeypatch, tmp_path)
    mode.state_file = tmp_path / "not-a-directory" / "mode.json"
    mode.state_file.parent.write_text("blocked")
    with pytest.raises(RuntimeError, match="Cannot save"):
        mode.set_enabled(False)
    assert mode.status()["enabled"]
    assert listener._take_external_feedback() is None


def test_http_mode_controls_and_status(monkeypatch, tmp_path):
    mode, listener, calls = setup_mode(monkeypatch, tmp_path)
    server = VoiceControlServer("127.0.0.1", 0, mode.callbacks(), status_callback=mode.status)
    server.start()
    host, port = server.address
    try:
        for route, expected, count in [("pause", False, 2), ("resume", True, 1), ("toggle", False, 2)]:
            with urlopen(Request(f"http://{host}:{port}/voice/{route}", method="POST"), timeout=2) as response:
                assert response.status == 202
            drain(listener)
            with urlopen(f"http://{host}:{port}/voice/status", timeout=2) as response:
                assert response.headers["Content-Type"] == "application/json"
                assert json.load(response)["enabled"] is expected
            assert calls[-1] == count
    finally:
        server.stop()


def test_old_audio_cannot_be_retagged_after_fast_pause_resume(monkeypatch, tmp_path):
    mode, listener, calls = setup_mode(monkeypatch, tmp_path)
    listener._recognition_context.epoch = listener.input_epoch
    mode.set_enabled(False)
    mode.set_enabled(True)
    result = mode.recognize(lambda samples: calls.append("stale"), None)
    assert result["kind"] == "ignored"
    assert calls == []
