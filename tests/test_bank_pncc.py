"""Optional PNCC tests use synthetic input only; install feature PR requirements."""
import numpy as np
import pytest

pytest.importorskip('spafe')
pytest.importorskip('scipy')
from ha_voice.bank_pncc import extract_pncc


@pytest.mark.parametrize('length', [1, 400, 16000])
def test_silent_channels_are_finite_without_hiding_nans(length):
    features = extract_pncc(np.zeros(length, dtype=np.float32))
    assert features.shape[1] == 26
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    np.testing.assert_array_equal(features, 0.)


@pytest.mark.parametrize('audio', [[], [np.nan], [np.inf], [[0, 1]]])
def test_invalid_audio_is_rejected(audio):
    with pytest.raises(ValueError, match='Finite non-empty mono'):
        extract_pncc(audio)


def test_quiet_modulated_synthetic_audio_retains_finite_temporal_features():
    t = np.arange(16000) / 16000
    audio = .0001 * np.sin(2 * np.pi * (220 * t + 80 * t**2)) * np.sin(np.pi * t)**2
    features = extract_pncc(audio)
    assert features.shape == (99, 26)
    assert np.isfinite(features).all()
    assert np.linalg.norm(features[:, 13:]) > 0
