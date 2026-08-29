"""Two-stage start-phrase gate for continuous command recognition."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import numpy as np

from .audio import trim_spoken_phrase
from .features import extract_command_features, extract_mfcc
from .matcher import Template, classify


class StartPhraseGate:
    """Wait for a start phrase, then accept exactly one command utterance."""

    def __init__(
        self,
        *,
        sample_rate: int,
        start_label: str,
        display_name: str,
        templates: list[Template],
        max_distance: float,
        min_margin: float,
        top_k: int,
        command_timeout_seconds: float,
        min_audio_seconds: float,
        max_audio_seconds: float,
        min_command_audio_seconds: float,
        command_matcher: Callable[[np.ndarray], dict[str, Any]],
    ) -> None:
        self.sample_rate = sample_rate
        self.start_label = start_label
        self.display_name = display_name
        self.templates = templates
        self.max_distance = max_distance
        self.min_margin = min_margin
        self.top_k = top_k
        self.command_timeout_seconds = command_timeout_seconds
        self.min_audio_seconds = min_audio_seconds
        self.max_audio_seconds = max_audio_seconds
        self.min_command_audio_seconds = min_command_audio_seconds
        self.command_matcher = command_matcher
        self._command_deadline = 0.0

    @property
    def phase(self) -> str:
        if self._command_deadline > time.monotonic():
            return "awaiting_command"
        self._command_deadline = 0.0
        return "waiting_for_start"

    def arm_command_window(self) -> None:
        """Start a fresh full command window, including after feedback playback."""
        self._command_deadline = time.monotonic() + self.command_timeout_seconds

    def __call__(self, samples: np.ndarray) -> dict[str, Any]:
        trimmed = trim_spoken_phrase(samples, self.sample_rate)
        if trimmed.size < self.sample_rate // 5:
            raise ValueError("No clear speech was detected")
        audio_seconds = trimmed.size / self.sample_rate
        if self.phase == "awaiting_command":
            if audio_seconds < self.min_command_audio_seconds:
                self._command_deadline = 0.0
                return {
                    "kind": "command",
                    "accepted": False,
                    "command": None,
                    "utterance": None,
                    "best_command": None,
                    "best_utterance": None,
                    "score": 0.0,
                    "margin": 0.0,
                    "scores": {},
                    "rejection_reason": "command_too_short",
                    "min_audio_seconds": self.min_command_audio_seconds,
                }
            return self._process_command_features(
                extract_command_features(trimmed, self.sample_rate)
            )
        if audio_seconds < self.min_audio_seconds:
            return {
                "kind": "ignored",
                "accepted": False,
                "command": None,
                "utterance": None,
                "best_command": None,
                "best_utterance": None,
                "score": 0.0,
                "margin": 0.0,
                "scores": {},
                "rejection_reason": "start_too_short",
                "min_audio_seconds": self.min_audio_seconds,
            }
        if audio_seconds > self.max_audio_seconds:
            return {
                "kind": "ignored",
                "accepted": False,
                "command": None,
                "utterance": None,
                "best_command": None,
                "best_utterance": None,
                "score": 0.0,
                "margin": 0.0,
                "scores": {},
                "rejection_reason": "start_too_long",
                "max_audio_seconds": self.max_audio_seconds,
            }
        features = extract_mfcc(trimmed, self.sample_rate)
        return self.process_features(features)

    def _process_command_features(self, features: np.ndarray) -> dict[str, Any]:
        self._command_deadline = 0.0
        result = self.command_matcher(features)
        result["kind"] = "command"
        return result

    def process_features(self, features: np.ndarray) -> dict[str, Any]:
        if self.phase == "awaiting_command":
            return self._process_command_features(features)

        result = classify(
            features,
            self.templates,
            max_distance=self.max_distance,
            min_margin=self.min_margin,
            top_k=self.top_k,
            template_limits={self.start_label: 8, "_not_start_phrase": 8},
        )
        is_start = result.accepted and result.command == self.start_label
        best_label = min(result.per_command, key=result.per_command.get)
        if is_start:
            self.arm_command_window()
        return {
            "kind": "start_phrase" if is_start else "ignored",
            "accepted": is_start,
            "command": None,
            "utterance": self.display_name if is_start else None,
            "best_command": best_label,
            "best_utterance": self.display_name if best_label == self.start_label else None,
            "score": result.score,
            "margin": result.margin,
            "scores": result.per_command,
        }
