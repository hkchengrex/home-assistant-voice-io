from pathlib import Path

import numpy as np

from ha_voice.matcher import Template
from ha_voice.start_phrase import StartPhraseGate


def _features(offset: float = 0.0) -> np.ndarray:
    base = np.linspace(-1, 1, 30, dtype=np.float32)[:, None]
    return np.concatenate((base, base * 0.5), axis=1) + offset


def _gate() -> StartPhraseGate:
    start = _features()
    templates = [
        Template("_start_phrase", Path("start-1.wav"), start + 0.01),
        Template("_start_phrase", Path("start-2.wav"), start - 0.01),
        Template("_not_start_phrase", Path("negative.wav"), start + 2.0),
    ]

    def command_matcher(features: np.ndarray) -> dict[str, object]:
        del features
        return {"accepted": True, "command": "lights_off"}

    return StartPhraseGate(
        sample_rate=16000,
        start_label="_start_phrase",
        display_name="Start phrase",
        templates=templates,
        max_distance=0.5,
        min_margin=0.1,
        top_k=2,
        command_timeout_seconds=5.0,
        min_audio_seconds=0.56,
        max_audio_seconds=1.35,
        min_command_audio_seconds=0.42,
        command_matcher=command_matcher,
    )


def test_gate_opens_for_start_phrase_then_accepts_one_command() -> None:
    gate = _gate()

    start_result = gate.process_features(_features())
    command_result = gate.process_features(_features(0.5))

    assert start_result["kind"] == "start_phrase"
    assert start_result["accepted"] is True
    assert gate.phase == "waiting_for_start"
    assert command_result["kind"] == "command"
    assert command_result["command"] == "lights_off"


def test_gate_ignores_non_start_speech() -> None:
    gate = _gate()

    result = gate.process_features(_features(2.0))

    assert result["kind"] == "ignored"
    assert result["accepted"] is False
    assert gate.phase == "waiting_for_start"


def test_command_window_can_be_rearmed_after_spoken_feedback(monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr("ha_voice.start_phrase.time.monotonic", lambda: now[0])
    gate = _gate()

    gate.process_features(_features())
    now[0] = 14.0
    gate.arm_command_window()
    now[0] = 18.0

    assert gate.phase == "awaiting_command"


def test_armed_gate_extracts_command_features_from_audio() -> None:
    gate = _gate()
    received: list[np.ndarray] = []
    gate.command_matcher = lambda features: (
        received.append(features) or {"accepted": True, "command": "lights_off"}
    )
    time_axis = np.arange(16000, dtype=np.float32) / 16000
    samples = 0.2 * np.sin(2 * np.pi * 220 * time_axis)
    gate.arm_command_window()

    result = gate(samples)

    assert result["kind"] == "command"
    assert result["command"] == "lights_off"
    assert received[0].shape[1] == 26


def test_gate_rejects_overlong_start_audio_but_not_commands() -> None:
    gate = _gate()
    time_axis = np.arange(round(1.7 * 16000), dtype=np.float32) / 16000
    samples = 0.2 * np.sin(2 * np.pi * 220 * time_axis)

    start_result = gate(samples)
    gate.arm_command_window()
    command_result = gate(samples)

    assert start_result["kind"] == "ignored"
    assert start_result["rejection_reason"] == "start_too_long"
    assert command_result["kind"] == "command"


def test_gate_rejects_short_start_audio_but_not_commands() -> None:
    gate = _gate()
    time_axis = np.arange(round(0.54 * 16000), dtype=np.float32) / 16000
    samples = 0.2 * np.sin(2 * np.pi * 220 * time_axis)

    start_result = gate(samples)
    gate.arm_command_window()
    command_result = gate(samples)

    assert start_result["kind"] == "ignored"
    assert start_result["rejection_reason"] == "start_too_short"
    assert start_result["min_audio_seconds"] == 0.56
    assert command_result["kind"] == "command"


def test_gate_rejects_implausibly_short_command_audio() -> None:
    gate = _gate()
    time_axis = np.arange(round(0.38 * 16000), dtype=np.float32) / 16000
    samples = 0.2 * np.sin(2 * np.pi * 220 * time_axis)
    gate.arm_command_window()

    result = gate(samples)

    assert result["kind"] == "command"
    assert result["accepted"] is False
    assert result["rejection_reason"] == "command_too_short"
    assert result["min_audio_seconds"] == 0.42
    assert gate.phase == "waiting_for_start"
