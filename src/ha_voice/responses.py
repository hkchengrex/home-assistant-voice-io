"""Preloaded randomized spoken responses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np

from .audio import load_wav


@dataclass(frozen=True)
class ResponseClip:
    path: Path
    samples: np.ndarray
    sample_rate: int


Playback = Callable[[np.ndarray, int, int | str | None], None]


def _sounddevice_playback(samples: np.ndarray, sample_rate: int, device: int | str | None) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install the 'capture' extra to play voice responses") from exc
    sd.play(samples, samplerate=sample_rate, device=device)
    sd.wait()


class VoiceResponsePlayer:
    """Load configured response groups once and avoid immediate repeats."""

    def __init__(
        self,
        assets_dir: Path,
        groups: Mapping[str, str],
        *,
        device: int | str | None = None,
        playback_sample_rate: int = 24_000,
        rng: random.Random | random.SystemRandom | None = None,
        playback: Playback = _sounddevice_playback,
    ) -> None:
        self.device = device
        self.playback = playback
        self.rng = rng or random.SystemRandom()
        self.groups: dict[str, tuple[ResponseClip, ...]] = {}
        self._last_path: dict[str, Path] = {}
        for group, prefix in groups.items():
            clips = []
            for path in sorted(assets_dir.glob(f"{prefix}_*.wav")):
                audio = load_wav(path, playback_sample_rate)
                clips.append(ResponseClip(path, audio.samples, audio.sample_rate))
            if not clips:
                raise ValueError(f"No voice responses found for '{group}' in {assets_dir}")
            self.groups[group] = tuple(clips)

    def play(self, group: str) -> Path:
        if group not in self.groups:
            raise ValueError(f"No voice response is configured for '{group}'")
        clips = self.groups[group]
        previous = self._last_path.get(group)
        choices = [clip for clip in clips if len(clips) == 1 or clip.path != previous]
        clip = self.rng.choice(choices)
        self.playback(clip.samples, clip.sample_rate, self.device)
        self._last_path[group] = clip.path
        return clip.path
