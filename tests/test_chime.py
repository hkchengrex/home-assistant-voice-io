import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ha_voice.chime import confirmation_chime_samples, play_confirmation_chime


def test_confirmation_chime_is_short_and_bounded() -> None:
    samples = confirmation_chime_samples(16000)

    assert samples.dtype == np.float32
    assert 0.2 < samples.size / 16000 < 0.3
    assert 0.1 < float(np.max(np.abs(samples))) <= 0.16


def test_confirmation_chime_rejects_invalid_sample_rate() -> None:
    with pytest.raises(ValueError, match="positive"):
        confirmation_chime_samples(0)


def test_confirmation_chime_uses_output_native_rate(monkeypatch) -> None:
    played = {}
    fake_sounddevice = SimpleNamespace(
        query_devices=lambda device, kind: {"default_samplerate": 48000},
        play=lambda samples, **kwargs: played.update(samples=samples, **kwargs),
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    play_confirmation_chime(sample_rate=16000, device=2)

    assert played["samplerate"] == 48000
    assert played["device"] == 2
    assert played["blocking"] is True
    assert 0.2 < played["samples"].size / 48000 < 0.3
