from io import BytesIO
from pathlib import Path
import wave

import numpy as np
import pytest

from ha_voice.audio import Audio, load_wav
from ha_voice.config import load_config
from ha_voice.studio_server import (
    capture_cued_recording,
    has_command_recordings,
    input_devices,
    match_recording,
    match_features,
    recording_file,
    recording_files,
    replace_recording,
    state_payload,
    store_recording,
)
from ha_voice.matcher import MatchResult


ROOT = Path(__file__).parents[1]


def test_match_features_uses_distinct_action_for_margin(monkeypatch) -> None:
    config = load_config(ROOT / "commands.toml")
    monkeypatch.setattr(
        "ha_voice.studio_server.classify",
        lambda *args, **kwargs: MatchResult(
            command=None,
            score=3.2,
            runner_up_score=3.25,
            margin=0.015,
            accepted=False,
            per_command={
                "byebye": 3.2,
                "lights_off": 3.25,
                "lights_on": 3.5,
                "day_routine": 3.6,
                "good_night": 3.8,
            },
        ),
    )

    result = match_features(
        features=np.ones((5, 26), dtype=np.float32),
        config=config,
        templates=[],
    )

    assert result["accepted"] is True
    assert result["command"] == "byebye"
    assert result["runner_command"] == "lights_on"
    assert result["runner_score"] == 3.5
    assert result["margin"] == pytest.approx((3.5 - 3.2) / 3.5)


def test_match_features_rejects_close_different_action(monkeypatch) -> None:
    config = load_config(ROOT / "commands.toml")
    monkeypatch.setattr(
        "ha_voice.studio_server.classify",
        lambda *args, **kwargs: MatchResult(
            command=None,
            score=3.2,
            runner_up_score=3.25,
            margin=0.015,
            accepted=False,
            per_command={
                "byebye": 3.2,
                "lights_on": 3.25,
                "lights_off": 3.4,
                "day_routine": 3.6,
                "good_night": 3.8,
            },
        ),
    )

    result = match_features(
        features=np.ones((5, 26), dtype=np.float32),
        config=config,
        templates=[],
    )

    assert result["accepted"] is False
    assert result["command"] is None
    assert result["best_command"] == "byebye"
    assert result["runner_command"] == "lights_on"


def test_match_features_rejects_command_close_to_reviewed_negative(
    monkeypatch,
) -> None:
    config = load_config(ROOT / "commands.toml")
    monkeypatch.setattr(
        "ha_voice.studio_server.classify",
        lambda *args, **kwargs: MatchResult(
            command="_not_command",
            score=3.1,
            runner_up_score=3.2,
            margin=0.03,
            accepted=False,
            per_command={
                "good_night": 3.2,
                "lights_on": 3.8,
                "_not_command": 3.1,
            },
        ),
    )

    result = match_features(
        features=np.ones((5, 26), dtype=np.float32),
        config=config,
        templates=[],
    )

    assert result["accepted"] is False
    assert result["best_command"] == "good_night"
    assert result["runner_command"] == "_not_command"
    assert result["margin"] == 0.0


def test_input_devices_marks_only_physical_sources_for_comparison(monkeypatch) -> None:
    monkeypatch.setattr(
        "ha_voice.studio_server.list_devices",
        lambda: [
            {"name": "default", "max_input_channels": 32},
            {"name": "Built-in Audio", "max_input_channels": 2},
            {"name": "speaker.monitor", "max_input_channels": 2},
            {"name": "USB microphone", "max_input_channels": 1},
        ],
    )

    devices = input_devices()

    assert [item["name"] for item in devices] == [
        "default",
        "Built-in Audio",
        "speaker.monitor",
        "USB microphone",
    ]
    assert [item["comparison_eligible"] for item in devices] == [
        False,
        True,
        False,
        True,
    ]


def test_input_devices_prefers_routed_sources_over_hardware_aliases(monkeypatch) -> None:
    monkeypatch.setattr(
        "ha_voice.studio_server.list_devices",
        lambda: [
            {"name": "Built-in (hw:0,0)", "max_input_channels": 2},
            {"name": "sysdefault", "max_input_channels": 128},
            {"name": "alsa_input.pci-built-in", "max_input_channels": 2},
            {"name": "alsa_input.usb-external", "max_input_channels": 1},
        ],
    )

    devices = input_devices()

    assert [item["comparison_eligible"] for item in devices] == [
        False,
        False,
        True,
        True,
    ]


def _speech_like_wav(sample_rate: int = 48000) -> bytes:
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    voiced = 0.25 * np.sin(2 * np.pi * 180 * time)
    samples = np.concatenate(
        (np.zeros(sample_rate // 2, dtype=np.float32), voiced, np.zeros(sample_rate // 2, dtype=np.float32))
    )
    encoded = BytesIO()
    with wave.open(encoded, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((samples * 32767).astype("<i2").tobytes())
    return encoded.getvalue()


def test_state_payload_includes_prompt_and_progress(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    command_dir = tmp_path / "good_night"
    command_dir.mkdir()
    (command_dir / "one.wav").write_bytes(b"placeholder")
    devices = [{"id": 3, "name": "Microphone Array", "channels": 2}]
    payload = state_payload(config, tmp_path, devices)
    command = next(item for item in payload["commands"] if item["name"] == "good_night")
    assert [item["utterance"] for item in payload["commands"]] == [
        "Lights on",
        "Lights off (original language)",
        "Good night",
        "Lights on (original language)",
        "byebye",
        "byebye (native)",
        "Nothing",
        "Anything except the configured commands",
        "Start phrase",
    ]
    assert config.commands["byebye_native"].intent_group == "all_lights_off"
    assert payload["devices"] == devices
    assert command["utterance"] == "Good night"
    assert command["target_samples"] == 10
    assert command["count"] == 1
    negative = payload["commands"][-2]
    assert negative["name"] == "_not_command"
    assert negative["is_negative"] is True
    start_phrase = payload["commands"][-1]
    assert start_phrase["name"] == "_start_phrase"
    assert start_phrase["target_samples"] == 15
    assert start_phrase["is_start_phrase"] is True
    assert payload["start_phrase_ready"] is False
    assert payload["start_phrase_enabled"] is True
    assert payload["command_timeout_seconds"] == 3.0


def test_store_recording_accepts_negative_calibration_sample(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    path = store_recording(
        wav_data=_speech_like_wav(),
        command_name="_not_command",
        config=config,
        recordings_dir=tmp_path,
    )
    assert path.parent == tmp_path / "_not_command"


def test_store_recording_accepts_start_phrase_sample(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    path = store_recording(
        wav_data=_speech_like_wav(),
        command_name="_start_phrase",
        config=config,
        recordings_dir=tmp_path,
    )
    assert path.parent == tmp_path / "_start_phrase"


def test_cued_recording_chimes_then_captures_and_saves(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_config(ROOT / "commands.toml")
    events = []
    time = np.arange(16000, dtype=np.float32) / 16000
    captured = Audio(
        samples=(0.2 * np.sin(2 * np.pi * 180 * time)).astype(np.float32),
        sample_rate=16000,
    )
    monkeypatch.setattr(
        "ha_voice.studio_server.play_confirmation_chime",
        lambda **kwargs: events.append(("chime", kwargs)),
    )
    monkeypatch.setattr(
        "ha_voice.studio_server.record_audio",
        lambda **kwargs: events.append(("capture", kwargs)) or captured,
    )

    output = capture_cued_recording(
        command_name="good_night",
        device=7,
        config=config,
        recordings_dir=tmp_path,
    )

    assert [event[0] for event in events] == ["chime", "capture"]
    assert events[1][1] == {
        "seconds": 3.0,
        "sample_rate": 16000,
        "device": 7,
        "countdown": 0,
        "trim": False,
    }
    assert output.parent == tmp_path / "good_night"
    assert output.exists()


def test_command_recording_check_ignores_start_phrase_only(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    start_dir = tmp_path / config.start_phrase.name
    start_dir.mkdir(parents=True)
    (start_dir / "start.wav").write_bytes(b"start")

    assert has_command_recordings(config, tmp_path) is False

    command_dir = tmp_path / "good_night"
    command_dir.mkdir()
    (command_dir / "command.wav").write_bytes(b"command")
    assert has_command_recordings(config, tmp_path) is True


def test_replace_recording_keeps_count_and_removes_selected_take(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    previous = store_recording(
        wav_data=_speech_like_wav(),
        command_name="_start_phrase",
        config=config,
        recordings_dir=tmp_path,
    )

    replacement = replace_recording(
        wav_data=_speech_like_wav(),
        command_name="_start_phrase",
        filename=previous.name,
        config=config,
        recordings_dir=tmp_path,
    )

    assert not previous.exists()
    assert replacement.exists()
    assert len(recording_files("_start_phrase", config, tmp_path)) == 1


def test_store_recording_validates_trims_and_resamples(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    path = store_recording(
        wav_data=_speech_like_wav(),
        command_name="good_night",
        config=config,
        recordings_dir=tmp_path,
    )
    assert path.parent == tmp_path / "good_night"
    audio = load_wav(path)
    assert audio.sample_rate == 16000
    assert 0.9 < audio.samples.size / audio.sample_rate < 1.4


def test_store_recording_rejects_unknown_command(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    with pytest.raises(ValueError, match="Unknown command"):
        store_recording(
            wav_data=_speech_like_wav(),
            command_name="open_front_door",
            config=config,
            recordings_dir=tmp_path,
        )


def test_match_recording_identifies_saved_command(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    wav_data = _speech_like_wav()
    store_recording(
        wav_data=wav_data,
        command_name="good_night",
        config=config,
        recordings_dir=tmp_path,
    )

    result = match_recording(
        wav_data=wav_data,
        config=config,
        recordings_dir=tmp_path,
    )

    assert result["accepted"] is True
    assert result["command"] == "good_night"
    assert result["utterance"] == "Good night"


def test_saved_recordings_can_be_listed_and_resolved(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    command_dir = tmp_path / "good_night"
    command_dir.mkdir()
    first = command_dir / "one.wav"
    second = command_dir / "two.wav"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    assert recording_files("good_night", config, tmp_path) == [first, second]
    assert recording_file("good_night", "two.wav", config, tmp_path) == second


def test_saved_recording_rejects_path_traversal(tmp_path: Path) -> None:
    config = load_config(ROOT / "commands.toml")
    with pytest.raises(ValueError, match="Invalid recording name"):
        recording_file("good_night", "../secret.wav", config, tmp_path)
