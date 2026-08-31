from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

import numpy as np
import pytest

from ha_voice.audio import load_wav
from ha_voice.diagnostics import TriggerCaptureQueue
from ha_voice.config import load_config


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


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "commands.toml")


def test_missed_attempts_opt_in_is_shared_and_expires(tmp_path, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr("ha_voice.diagnostics.time.time", lambda: clock[0])
    studio = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    listener = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    rejected = {"kind": "ignored", "accepted": False, "rejection_reason": "start_too_short"}
    assert listener.capture(_samples(), rejected) is None
    assert studio.set_review_mode(True) == {"active": True, "remaining_seconds": 300}
    event = listener.capture(_samples(), rejected)
    assert studio.list_events()[0]["attempt"]["audio_url"].endswith("/attempt.wav")
    assert studio.audio_path(event, "attempt.wav").exists()
    clock[0] += 300
    assert not listener.review_status()["active"]
    assert listener.capture(_samples(), rejected) is None
    studio.set_review_mode(True)
    studio.set_review_mode(False)
    assert listener.capture(_samples(), rejected) is None
    assert len(studio.list_events()) == 1


@pytest.mark.parametrize("policy", ["broken", "null", "[]", '{"started_at": 0, "expires_at": 99999999999}', '{"started_at": 0, "expires_at": NaN}'])
def test_invalid_review_policy_fails_closed(tmp_path, policy):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    (queue.root.parent / "review-mode.json").write_text(policy)
    assert not queue.review_status()["active"]


def test_misses_are_bounded_and_do_not_attach_to_stale_wake(tmp_path):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000, max_events=2)
    queue.capture(_samples(), _start_result())
    queue.capture(_samples(), {"kind": "ignored", "accepted": False})
    assert queue.capture(_samples(), _command_result()) is None
    queue.set_review_mode(True)
    for _ in range(4):
        queue.capture(_samples(), {"kind": "ignored", "accepted": False})
    assert len(queue.list_events()) == 2
    assert all(event.get("attempt") for event in queue.list_events())


def test_rejected_command_is_saved_without_review_mode(tmp_path):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event = queue.capture(_samples(), _start_result())
    assert queue.capture(_samples(), {**_command_result(), "accepted": False}) == event
    assert queue.list_events()[0]["command"]["accepted"] is False


def test_correct_command_does_not_reject_valid_wake(tmp_path, config):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event = queue.capture(_samples(), _start_result())
    queue.capture(_samples(), {**_command_result(), "accepted": False})
    original_start = queue.audio_path(event, "start.wav").read_bytes()
    original_command = queue.audio_path(event, "command.wav").read_bytes()
    result = queue.teach(event, "command", "lights_on", config)
    assert not result["already_taught"]
    assert (tmp_path / "lights_on" / result["filename"]).exists()
    assert not (tmp_path / "_not_start_phrase").exists()
    assert not (tmp_path / "_not_command").exists()
    assert queue.audio_path(event, "start.wav").read_bytes() == original_start
    assert queue.audio_path(event, "command.wav").read_bytes() == original_command
    assert queue.list_events()[0]["command"]["taught_as"] == "lights_on"
    with pytest.raises(ValueError, match="separately"):
        queue.promote_false_trigger(event)


def test_missed_wake_can_be_taught_and_provenance_survives_dismiss(tmp_path, config):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    queue.set_review_mode(True)
    event = queue.capture(_samples(), {"kind": "ignored", "accepted": False})
    result = queue.teach(event, "attempt", config.start_phrase.name, config)
    queue.dismiss(event)
    assert (tmp_path / config.start_phrase.name / result["filename"]).exists()
    provenance = json.loads((tmp_path / "_review_metadata" / f"{event}_attempt.json").read_text())
    assert provenance["event"]["attempt"]["accepted"] is False


def test_negative_label_applies_to_only_one_clip(tmp_path, config):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event = queue.capture(_samples(), _start_result())
    queue.capture(_samples(), _command_result())
    queue.teach(event, "start", "_not_start_phrase", config)
    assert len(list((tmp_path / "_not_start_phrase").glob("*.wav"))) == 1
    assert not (tmp_path / "_not_command").exists()
    assert "taught_as" not in queue.list_events()[0]["command"]


def test_duplicate_concurrent_labels_do_not_duplicate_audio(tmp_path, config):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event = queue.capture(_samples(), _start_result())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: queue.teach(event, "start", config.start_phrase.name, config), range(2)))
    assert sorted(r["already_taught"] for r in results) == [False, True]
    assert len(list((tmp_path / config.start_phrase.name).glob("*.wav"))) == 1
    with pytest.raises(ValueError, match="already taught"):
        queue.teach(event, "start", "lights_off", config)


@pytest.mark.parametrize("clip,label", [("../command", "lights_on"), ("start", "../outside"), ("start", "unknown"), ("missing", "lights_on"), ("command", "lights_on")])
def test_invalid_labels_or_missing_clips_cannot_write_templates(tmp_path, config, clip, label):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event = queue.capture(_samples(), _start_result())
    with pytest.raises((ValueError, FileNotFoundError)):
        queue.teach(event, clip, label, config)
    assert not (tmp_path / "lights_on").exists()


def test_failed_metadata_save_rolls_back_training_take(tmp_path, config, monkeypatch):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event = queue.capture(_samples(), _start_result())
    def fail(*args):
        raise OSError("Disk full")
    monkeypatch.setattr(queue, "_write_json", fail)
    with pytest.raises(OSError):
        queue.teach(event, "start", config.start_phrase.name, config)
    assert not list((tmp_path / config.start_phrase.name).glob("*.wav"))
    assert queue.audio_path(event, "start.wav").exists()


def test_existing_template_is_never_overwritten(tmp_path, config):
    queue = TriggerCaptureQueue(tmp_path, sample_rate=16000)
    event = queue.capture(_samples(), _start_result())
    destination = tmp_path / "lights_on" / f"{event}_start.wav"
    destination.parent.mkdir()
    destination.write_bytes(b"existing")
    with pytest.raises(ValueError, match="already exists"):
        queue.teach(event, "start", "lights_on", config)
    assert destination.read_bytes() == b"existing"
