import sys
import threading
import time
import types

import numpy as np

from ha_voice.continuous import ContinuousListener, VoiceSegmenter


def _blocks(value: float, count: int, block_size: int = 320) -> list[np.ndarray]:
    return [np.full(block_size, value, dtype=np.float32) for _ in range(count)]


def test_segmenter_emits_after_trailing_pause() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    stream = _blocks(0.001, 10) + _blocks(0.08, 30) + _blocks(0.001, 15)
    for block in stream:
        output.extend(segmenter.process(block))

    assert len(output) == 1
    assert 0.5 < output[0].samples.size / segmenter.sample_rate <= 1.0
    assert output[0].hit_duration_limit is False
    assert not segmenter.active


def test_segmenter_ignores_single_noise_spike() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    stream = _blocks(0.001, 10) + _blocks(0.08, 1) + _blocks(0.001, 40)
    for block in stream:
        output.extend(segmenter.process(block))

    assert output == []
    assert not segmenter.active


def test_segmenter_emits_multiple_commands() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    utterance = _blocks(0.08, 25) + _blocks(0.001, 30)
    for block in _blocks(0.001, 10) + utterance + utterance:
        output.extend(segmenter.process(block))

    assert len(output) == 2


def test_segmenter_calibrates_to_steady_room_noise() -> None:
    segmenter = VoiceSegmenter(
        min_rms=0.006,
        noise_multiplier=1.7,
        calibration_ms=1000,
    )
    output: list[np.ndarray] = []
    for block in _blocks(0.007, 100):
        output.extend(segmenter.process(block))

    assert output == []
    assert not segmenter.active
    assert segmenter.noise_rms > 0.006


def test_segmenter_hears_speech_after_room_calibration() -> None:
    segmenter = VoiceSegmenter(
        min_rms=0.006,
        noise_multiplier=1.7,
        calibration_ms=1000,
    )
    output: list[np.ndarray] = []
    stream = _blocks(0.007, 50) + _blocks(0.05, 30) + _blocks(0.001, 15)
    for block in stream:
        output.extend(segmenter.process(block))

    assert len(output) == 1


def test_segmenter_caps_noisy_utterance_at_two_and_a_half_seconds() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    for block in _blocks(0.08, 140):
        output.extend(segmenter.process(block))

    assert len(output) == 1
    assert output[0].samples.size / segmenter.sample_rate <= 2.5
    assert output[0].hit_duration_limit is True


def test_segmenter_pre_roll_recovers_quiet_first_word() -> None:
    segmenter = VoiceSegmenter(
        min_rms=0.006,
        noise_multiplier=1.7,
        pre_roll_ms=700,
    )
    output: list[np.ndarray] = []
    stream = (
        _blocks(0.001, 10)
        + _blocks(0.005, 15)
        + _blocks(0.001, 5)
        + _blocks(0.05, 10)
        + _blocks(0.001, 15)
    )
    for block in stream:
        output.extend(segmenter.process(block))

    assert len(output) == 1
    assert np.any(np.isclose(output[0].samples, 0.005))


class _FakeInputStream:
    """Stand-in for sounddevice.InputStream that optionally delivers audio."""

    def __init__(self, *, deliver: bool, blocksize: int, callback, **_: object) -> None:
        self.deliver = deliver
        self.blocksize = blocksize
        self.callback = callback
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._feed, daemon=True)

    def _feed(self) -> None:
        block = np.zeros((self.blocksize, 1), dtype=np.float32)
        while not self._stop.wait(0.02):
            self.callback(block, self.blocksize, None, None)

    def __enter__(self) -> "_FakeInputStream":
        if self.deliver:
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()


def _fake_sounddevice(monkeypatch, *, deliver: bool) -> None:
    module = types.ModuleType("sounddevice")
    module.InputStream = lambda **kwargs: _FakeInputStream(deliver=deliver, **kwargs)
    monkeypatch.setitem(sys.modules, "sounddevice", module)


def _wait_until_stopped(listener: ContinuousListener, seconds: float) -> dict:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        snapshot = listener.snapshot()
        if not snapshot["running"]:
            return snapshot
        time.sleep(0.02)
    return listener.snapshot()


def test_listener_stops_when_microphone_goes_silent(monkeypatch) -> None:
    _fake_sounddevice(monkeypatch, deliver=False)
    listener = ContinuousListener(heartbeat_timeout_seconds=None, audio_timeout_seconds=0.3)
    listener.start(device=None, matcher=lambda samples: {})

    snapshot = _wait_until_stopped(listener, 3.0)
    listener.stop()

    assert snapshot["running"] is False
    assert "no audio" in snapshot["error"]


def test_listener_keeps_running_while_audio_arrives(monkeypatch) -> None:
    _fake_sounddevice(monkeypatch, deliver=True)
    listener = ContinuousListener(heartbeat_timeout_seconds=None, audio_timeout_seconds=0.3)
    listener.start(device=None, matcher=lambda samples: {})

    snapshot = _wait_until_stopped(listener, 1.0)
    listener.stop()

    assert snapshot["running"] is True
    assert snapshot["error"] is None


def test_feedback_replacement_interrupts_active_and_discards_stale_pending() -> None:
    listener = ContinuousListener()
    listener._state["running"] = True
    calls = []
    active_cancel = threading.Event()
    stale_cancel = threading.Event()
    listener.enqueue_feedback(lambda: calls.append("active"), cancel=active_cancel.set)
    active = listener._take_external_feedback()
    assert active is not None
    listener.enqueue_feedback(lambda: calls.append("stale"), cancel=stale_cancel.set)
    listener.enqueue_feedback(lambda: calls.append("latest"), replace_pending=True)
    assert active_cancel.is_set()
    assert stale_cancel.is_set()
    latest = listener._take_external_feedback()
    assert latest is not None
    latest.callback()
    assert calls == ["latest"]
    assert listener._take_external_feedback() is None


def test_feedback_default_still_queues_fifo() -> None:
    listener = ContinuousListener()
    listener._state["running"] = True
    calls = []
    for i in range(3):
        listener.enqueue_feedback(lambda i=i: calls.append(i))
    for _ in range(3):
        item = listener._take_external_feedback()
        assert item is not None
        item.callback()
    assert calls == [0, 1, 2]


def test_paused_capture_stops_stream_but_keeps_control_feedback(monkeypatch):
    streams = []
    class Stream(_FakeInputStream):
        def __init__(self, **kwargs):
            super().__init__(deliver=True, **kwargs)
            self.paused = False
            self.transitions = []
            streams.append(self)
        def stop(self):
            self.paused = True
            self.transitions.append("stop")
        def start(self):
            self.paused = False
            self.transitions.append("start")
        def _feed(self):
            block = np.zeros((self.blocksize, 1), dtype=np.float32)
            while not self._stop.wait(0.005):
                if not self.paused:
                    self.callback(block, self.blocksize, None, None)
    module = types.ModuleType("sounddevice")
    module.InputStream = Stream
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    listener = ContinuousListener(heartbeat_timeout_seconds=None, audio_timeout_seconds=0.05)
    listener.set_input_paused(True)
    matched = []
    listener.start(device=None, matcher=lambda samples: matched.append(samples))
    try:
        deadline = time.monotonic() + 1
        while not listener.snapshot()["capture_paused"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert listener.snapshot()["capture_paused"]
        feedback = threading.Event()
        listener.enqueue_feedback(feedback.set)
        assert feedback.wait(1)
        time.sleep(0.1)  # Pausing must not trigger the missing-microphone watchdog.
        assert listener.snapshot()["running"]
        assert listener.snapshot()["phase"] == "paused"
        assert matched == []
        listener.set_input_paused(False)
        deadline = time.monotonic() + 1
        while listener.snapshot()["capture_paused"] and time.monotonic() < deadline:
            time.sleep(0.01)
        assert streams[0].transitions == ["stop", "start"]
        assert not listener.snapshot()["capture_paused"]
        assert listener.snapshot()["error"] is None
    finally:
        listener.stop()
