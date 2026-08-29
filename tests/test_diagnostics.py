from pathlib import Path

import numpy as np

from ha_voice.audio import load_wav
from ha_voice.diagnostics import TriggerCaptureQueue


def _samples(seconds: float = 0.7) -> np.ndarray:
    time = np.arange(round(16000 * seconds), dtype=np.float32) / 16000
    return (0.2 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)


def _start_result(score: float = 3.2) -> dict[str, object]:
    return {
        "kind": "start_phrase",
        "accepted": True,
        "score": score,
        "margin": 0.08,
        "audio_seconds": 0.7,
        "scores": {"_start_phrase": score, "_not_start_phrase": 3.8},
    }


def _command_result() -> dict[str, object]:
    return {
        "kind": "command",
        "accepted": True,
        "command": "good_night",
        "utterance": "Good night",
        "score": 3.1,
        "margin": 0.09,
        "audio_seconds": 0.6,
        "action_status": "sent",
    }


def test_trigger_queue_pairs_start_and_command_audio(tmp_path: Path) -> None:
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000, max_events=3)

    event_id = queue.capture(_samples(), _start_result())
    command_event_id = queue.capture(_samples(0.6), _command_result())

    assert command_event_id == event_id
    events = queue.list_events()
    assert len(events) == 1
    assert events[0]["id"] == event_id
    assert events[0]["start"]["audio_url"].endswith("/start.wav")
    assert events[0]["command"]["utterance"] == "Good night"
    assert events[0]["command"]["audio_url"].endswith("/command.wav")
    assert load_wav(queue.audio_path(event_id, "start.wav")).sample_rate == 16000


def test_trigger_queue_discards_oldest_event_at_capacity(tmp_path: Path) -> None:
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000, max_events=2)
    event_ids = [
        queue.capture(_samples(), _start_result(3.1 + index * 0.1))
        for index in range(3)
    ]

    listed_ids = [event["id"] for event in queue.list_events()]
    assert listed_ids == list(reversed(event_ids[1:]))
    assert not (queue.root / event_ids[0]).exists()


def test_promoting_false_trigger_creates_both_negative_templates(
    tmp_path: Path,
) -> None:
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event_id = queue.capture(_samples(), _start_result())
    queue.capture(_samples(0.6), _command_result())

    result = queue.promote_false_trigger(event_id)

    assert result["promoted"] == ["_not_start_phrase", "_not_command"]
    assert (tmp_path / "_not_start_phrase" / f"{event_id}_start.wav").exists()
    assert (tmp_path / "_not_command" / f"{event_id}_command.wav").exists()
    assert (tmp_path / "_negative_metadata" / f"{event_id}.json").exists()
    assert queue.list_events() == []


def test_dismiss_removes_candidate_without_training_it(tmp_path: Path) -> None:
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event_id = queue.capture(_samples(), _start_result())

    queue.dismiss(event_id)

    assert queue.list_events() == []
    assert not (tmp_path / "_not_start_phrase").exists()
