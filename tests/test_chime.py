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


def test_single_and_double_beeps_are_distinct_short_tones():
    from ha_voice.chime import beep_samples
    one = beep_samples(1)
    two = beep_samples(2)
    assert one.dtype == np.float32
    assert len(one) == 2160
    assert len(two) == 6480
    np.testing.assert_array_equal(two[:2160], one)
    assert not np.any(two[2160:4320])
    np.testing.assert_array_equal(two[4320:], one)
    assert float(np.max(np.abs(two))) <= 0.16
    with pytest.raises(ValueError):
        beep_samples(3)
