"""Configuration loading for the reusable voice pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import tomllib


_NAME = re.compile(r"[A-Za-z0-9_-]+")


@dataclass(frozen=True)
class RecognizerConfig:
    sample_rate: int = 16_000
    max_distance: float = 4.0
    min_margin: float = 0.12
    top_k: int = 3
    pcen_smoothing: float = 0.05
    pcen_alpha: float = 0.98

    @property
    def pcen_options(self) -> dict[str, float]:
        return {"smoothing": self.pcen_smoothing, "alpha": self.pcen_alpha}


@dataclass(frozen=True)
class CommandConfig:
    """Recognition metadata with no automation-platform dependency."""

    name: str
    description: str
    utterance: str
    target_samples: int
    intent_group: str
    response: str


@dataclass(frozen=True)
class CalibrationConfig:
    name: str = "_not_command"
    description: str = "Varied unrelated speech used only to tune rejection"
    utterance: str = "Anything except the configured commands"
    target_samples: int = 10


@dataclass(frozen=True)
class StartPhraseConfig:
    name: str = "_start_phrase"
    enabled: bool = False
    description: str = "Wake the listener before it accepts one command"
    utterance: str = "Start phrase"
    target_samples: int = 15
    max_distance: float = 3.8
    min_margin: float = 0.08
    top_k: int = 3
    command_timeout_seconds: float = 5.0
    min_audio_seconds: float = 0.2
    max_audio_seconds: float = 1.5
    min_command_audio_seconds: float = 0.2
    response: str = ""


@dataclass(frozen=True)
class AppConfig:
    recognizer: RecognizerConfig
    commands: dict[str, CommandConfig]
    calibration: CalibrationConfig
    start_phrase: StartPhraseConfig
    responses: dict[str, str]
    control_events: dict[str, str]


def _validate_name(value: str, field: str, *, allow_empty: bool = True) -> None:
    if not value and allow_empty:
        return
    if not _NAME.fullmatch(value):
        raise ValueError(f"{field} contains unsupported characters")


def load_config(path: Path) -> AppConfig:
    with path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    recognizer_raw = raw.get("recognizer", {})
    recognizer = RecognizerConfig(
        sample_rate=int(recognizer_raw.get("sample_rate", 16_000)),
        max_distance=float(recognizer_raw.get("max_distance", 4.0)),
        min_margin=float(recognizer_raw.get("min_margin", 0.12)),
        top_k=int(recognizer_raw.get("top_k", 3)),
        pcen_smoothing=float(recognizer_raw.get("pcen_smoothing", 0.05)),
        pcen_alpha=float(recognizer_raw.get("pcen_alpha", 0.98)),
    )
    if recognizer.sample_rate < 8_000:
        raise ValueError("recognizer.sample_rate must be at least 8000")
    if recognizer.max_distance <= 0:
        raise ValueError("recognizer.max_distance must be positive")
    if not 0 <= recognizer.min_margin < 1:
        raise ValueError("recognizer.min_margin must be between 0 and 1")
    if recognizer.top_k < 1:
        raise ValueError("recognizer.top_k must be at least 1")
    if not 0 < recognizer.pcen_smoothing <= 1:
        raise ValueError("recognizer.pcen_smoothing must be between 0 and 1")
    if not 0 <= recognizer.pcen_alpha <= 1:
        raise ValueError("recognizer.pcen_alpha must be between 0 and 1")

    responses = {str(name): str(prefix) for name, prefix in raw.get("responses", {}).items()}
    for name, prefix in responses.items():
        _validate_name(name, f"responses.{name}", allow_empty=False)
        _validate_name(prefix, f"responses.{name}", allow_empty=False)

    commands: dict[str, CommandConfig] = {}
    for name, command_raw in raw.get("commands", {}).items():
        _validate_name(name, f"commands.{name}", allow_empty=False)
        command = CommandConfig(
            name=name,
            description=str(command_raw.get("description", name)),
            utterance=str(command_raw.get("utterance", name.replace("_", " "))),
            target_samples=int(command_raw.get("target_samples", 10)),
            intent_group=str(command_raw.get("intent_group", name)),
            response=str(command_raw.get("response", "")),
        )
        if command.target_samples < 1:
            raise ValueError(f"commands.{name}.target_samples must be at least 1")
        _validate_name(command.intent_group, f"commands.{name}.intent_group", allow_empty=False)
        _validate_name(command.response, f"commands.{name}.response")
        if command.response and command.response not in responses:
            raise ValueError(f"commands.{name}.response references an unknown response group")
        commands[name] = command

    if not commands:
        raise ValueError("commands.toml must define at least one command")

    calibration_raw = raw.get("calibration", {})
    calibration = CalibrationConfig(
        description=str(calibration_raw.get("description", CalibrationConfig.description)),
        utterance=str(calibration_raw.get("utterance", CalibrationConfig.utterance)),
        target_samples=int(calibration_raw.get("target_samples", 10)),
    )
    if calibration.target_samples < 1:
        raise ValueError("calibration.target_samples must be at least 1")

    start_raw = raw.get("start_phrase", {})
    start_phrase = StartPhraseConfig(
        enabled=bool(start_raw.get("enabled", False)),
        description=str(start_raw.get("description", StartPhraseConfig.description)),
        utterance=str(start_raw.get("utterance", StartPhraseConfig.utterance)),
        target_samples=int(start_raw.get("target_samples", 15)),
        max_distance=float(start_raw.get("max_distance", 3.8)),
        min_margin=float(start_raw.get("min_margin", 0.08)),
        top_k=int(start_raw.get("top_k", 3)),
        command_timeout_seconds=float(start_raw.get("command_timeout_seconds", 5.0)),
        min_audio_seconds=float(start_raw.get("min_audio_seconds", 0.2)),
        max_audio_seconds=float(start_raw.get("max_audio_seconds", 1.5)),
        min_command_audio_seconds=float(start_raw.get("min_command_audio_seconds", 0.2)),
        response=str(start_raw.get("response", "")),
    )
    if start_phrase.target_samples < 1:
        raise ValueError("start_phrase.target_samples must be at least 1")
    if start_phrase.max_distance <= 0:
        raise ValueError("start_phrase.max_distance must be positive")
    if not 0 <= start_phrase.min_margin < 1:
        raise ValueError("start_phrase.min_margin must be between 0 and 1")
    if start_phrase.top_k < 1:
        raise ValueError("start_phrase.top_k must be at least 1")
    if start_phrase.command_timeout_seconds <= 0:
        raise ValueError("start_phrase.command_timeout_seconds must be positive")
    if start_phrase.min_audio_seconds <= 0 or start_phrase.max_audio_seconds <= 0:
        raise ValueError("start phrase audio durations must be positive")
    if start_phrase.min_command_audio_seconds <= 0:
        raise ValueError("start_phrase.min_command_audio_seconds must be positive")
    if start_phrase.min_audio_seconds >= start_phrase.max_audio_seconds:
        raise ValueError("start_phrase.min_audio_seconds must be less than max_audio_seconds")
    _validate_name(start_phrase.response, "start_phrase.response")
    if start_phrase.response and start_phrase.response not in responses:
        raise ValueError("start_phrase.response references an unknown response group")

    control_events: dict[str, str] = {}
    for route, response in raw.get("control_events", {}).items():
        route = str(route)
        response = str(response)
        if not route.startswith("/") or route == "/health" or "?" in route or "#" in route:
            raise ValueError(f"control_events.{route} is not a safe route")
        if response not in responses:
            raise ValueError(f"control_events.{route} references an unknown response group")
        control_events[route] = response

    return AppConfig(
        recognizer=recognizer,
        commands=commands,
        calibration=calibration,
        start_phrase=start_phrase,
        responses=responses,
        control_events=control_events,
    )
