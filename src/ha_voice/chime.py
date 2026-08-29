"""Short local audio feedback for the start-phrase gate."""

from __future__ import annotations

from typing import Any

import numpy as np


def confirmation_chime_samples(sample_rate: int = 16_000) -> np.ndarray:
    """Build a brief rising three-note chime with click-free edges."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    notes: list[np.ndarray] = []
    note_samples = round(sample_rate * 0.065)
    gap = np.zeros(round(sample_rate * 0.012), dtype=np.float32)
    time = np.arange(note_samples, dtype=np.float32) / sample_rate
    envelope = np.sin(np.linspace(0.0, np.pi, note_samples, dtype=np.float32)) ** 2
    for frequency in (659.25, 783.99, 987.77):
        tone = 0.16 * envelope * np.sin(2.0 * np.pi * frequency * time)
        notes.extend((tone.astype(np.float32), gap))
    notes.append(np.zeros(round(sample_rate * 0.025), dtype=np.float32))
    return np.concatenate(notes)


def play_confirmation_chime(
    *, sample_rate: int = 16_000, device: int | str | None = None
) -> None:
    """Play the confirmation chime through the selected local output."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install the sounddevice package to play audio") from exc
    playback_rate = sample_rate
    try:
        device_info: Any = sd.query_devices(device, "output")
        native_rate = round(float(device_info.get("default_samplerate", sample_rate)))
        if native_rate > 0:
            playback_rate = native_rate
    except (TypeError, ValueError, RuntimeError):
        pass
    sd.play(
        confirmation_chime_samples(playback_rate),
        samplerate=playback_rate,
        device=device,
        blocking=True,
    )
