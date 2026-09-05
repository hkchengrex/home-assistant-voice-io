"""Optional WebRTC speech detection; one instance per continuous stream."""

from __future__ import annotations

import numpy as np


class WebRtcDetector:
    """Accept mono float PCM frames without an additional energy gate."""

    def __init__(self, sample_rate: int = 16000, mode: int = 1) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError("WebRTC requires 8000, 16000, 32000, or 48000 Hz")
        if mode not in (0, 1, 2, 3):
            raise ValueError("WebRTC mode must be between 0 and 3")
        try:
            import webrtcvad
        except ImportError as exc:
            raise RuntimeError("Install the optional voice-io[vad] extra") from exc
        self.sample_rate = sample_rate
        self._vad = webrtcvad.Vad(mode)

    def __call__(self, samples: np.ndarray) -> bool:
        samples = np.asarray(samples, dtype=np.float32)
        valid_sizes = {self.sample_rate * ms // 1000 for ms in (10, 20, 30)}
        if samples.ndim != 1 or samples.size not in valid_sizes:
            raise ValueError("WebRTC needs one complete mono 10, 20, or 30 ms frame")
        if not np.isfinite(samples).all():
            raise ValueError("Audio must contain only finite values")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        return bool(self._vad.is_speech(pcm, self.sample_rate))
