from pathlib import Path

import pytest

import numpy as np

from ha_voice.audio import load_wav
from ha_voice.cli import (
    _format_listener_result,
    _parser,
    _recording_utterance,
    _save_training_sample,
)
from ha_voice.config import load_config


ROOT = Path(__file__).parents[1]


def test_format_listener_result_reports_start_phrase_scores() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "ignored",
        "score": 2.7,
        "margin": 0.06,
        "scores": {"_start_phrase": 3.2, "_not_start_phrase": 2.7},
    }

    message = _format_listener_result(result, config)

    assert message == "START: rejected start=3.200 rejection=2.700 margin=6.0%"


def test_format_listener_result_reports_duration_limit_rejection() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "ignored",
        "score": 0.0,
        "margin": 0.0,
        "audio_seconds": 2.5,
        "rejection_reason": "duration_limit",
    }

    assert _format_listener_result(result, config) == (
        "START: rejected reason=duration-limit audio=2.50s"
    )


def test_format_listener_result_reports_overlong_start_rejection() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "ignored",
        "audio_seconds": 1.76,
        "rejection_reason": "start_too_long",
        "max_audio_seconds": 1.35,
    }

    assert _format_listener_result(result, config) == (
        "START: rejected reason=too-long audio=1.76s max=1.35s"
    )


def test_format_listener_result_reports_short_start_rejection() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "ignored",
        "audio_seconds": 0.54,
        "rejection_reason": "start_too_short",
        "min_audio_seconds": 0.56,
    }

    assert _format_listener_result(result, config) == (
        "START: rejected reason=too-short audio=0.54s min=0.56s"
    )


def test_format_listener_result_reports_recognized_command() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "command",
        "accepted": True,
        "utterance": "Lights off",
        "score": 2.4,
        "margin": 0.18,
    }

    message = _format_listener_result(result, config)

    assert message == "COMMAND: Lights off score=2.400 margin=18.0% (no action)"


def test_format_listener_result_reports_short_command_rejection() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "command",
        "accepted": False,
        "audio_seconds": 0.38,
        "rejection_reason": "command_too_short",
        "min_audio_seconds": 0.42,
    }

    assert _format_listener_result(result, config) == (
        "COMMAND: rejected reason=too-short audio=0.38s min=0.42s (no action)"
    )


def test_format_listener_result_names_rejected_command_competitors() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "command",
        "accepted": False,
        "score": 3.2,
        "margin": 0.02,
        "best_command": "byebye",
        "runner_command": "lights_on",
        "runner_score": 3.265,
        "scores": {"byebye": 3.2, "lights_on": 3.265, "good_night": 3.8},
    }

    message = _format_listener_result(result, config)

    assert message == (
        "COMMAND: rejected best=byebye score=3.200 runner=lights_on "
        "runner_score=3.265 margin=2.0% (no action)"
    )


def test_format_listener_result_reports_sent_action() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "command",
        "accepted": True,
        "utterance": "Lights on",
        "score": 2.4,
        "margin": 0.18,
        "action_status": "sent",
    }

    assert _format_listener_result(result, config) == (
        "COMMAND: Lights on score=2.400 margin=18.0% action=sent"
    )


def test_format_listener_result_reports_failed_action() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "command",
        "accepted": True,
        "utterance": "Lights on",
        "score": 2.4,
        "margin": 0.18,
        "action_status": "failed",
        "action_error": "Automation backend is unavailable",
    }

    assert _format_listener_result(result, config) == (
        "COMMAND: Lights on score=2.400 margin=18.0% "
        "action=failed error=Automation backend is unavailable"
    )


def test_format_listener_result_includes_match_time_when_available() -> None:
    config = load_config(ROOT / "commands.toml")
    result = {
        "kind": "start_phrase",
        "score": 2.5,
        "margin": 0.12,
        "match_seconds": 0.237,
        "audio_seconds": 1.184,
    }

    message = _format_listener_result(result, config)

    assert message == (
        "START: recognized score=2.500 margin=12.0% match=0.24s audio=1.18s"
    )


def test_recording_prompt_supports_start_phrase_and_commands() -> None:
    config = load_config(ROOT / "commands.toml")

    assert _recording_utterance("_start_phrase", config) == "Start phrase"
    assert _recording_utterance("lights_off", config) == "Lights off (original language)"
    assert _recording_utterance("byebye", config) == "byebye"


def test_recording_prompt_rejects_unknown_set() -> None:
    config = load_config(ROOT / "commands.toml")

    with pytest.raises(SystemExit, match="Unknown recording set"):
        _recording_utterance("unknown", config)


def test_run_parser_supports_direct_command_testing() -> None:
    args = _parser().parse_args(
        [
            "run",
            "--direct",
            "--capture-command",
            "lights_off",
            "--capture-count",
            "5",
        ]
    )

    assert args.direct is True
    assert args.capture_command == "lights_off"
    assert args.capture_count == 5


def test_parser_supports_hugging_face_response_cloning() -> None:
    args = _parser().parse_args(
        [
            "clone-response",
            "--response",
            "welcome",
            "--text",
            "Welcome home",
            "--reference",
            "reference.wav",
            "--language",
            "English",
            "--confirm-upload",
        ]
    )

    assert args.response == "welcome"
    assert args.language == "English"
    assert args.space == "hkchengrex/OmniVoice"
    assert args.api_name == "/_clone_fn"
    assert args.token_env == "HF_TOKEN"
    assert args.attempts == 2
    assert args.confirm_upload is True


def test_save_training_sample_writes_listener_audio(tmp_path: Path) -> None:
    samples = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)

    path = _save_training_sample(
        samples,
        sample_rate=16000,
        command_name="lights_off",
        recordings=tmp_path,
    )

    audio = load_wav(path)
    assert path.parent == tmp_path / "lights_off"
    assert np.allclose(audio.samples, samples, atol=2 / 32767)
