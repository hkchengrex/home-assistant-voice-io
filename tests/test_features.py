import numpy as np

from ha_voice.features import extract_command_features, extract_mfcc, extract_pcen_cepstra


def _tone(frequency: float, seconds: float = 1.0, sample_rate: int = 16000) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.3 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)


def test_extract_mfcc_is_finite_and_has_expected_width() -> None:
    features = extract_mfcc(_tone(440))
    assert features.ndim == 2
    assert features.shape[0] > 50
    assert features.shape[1] == 26
    assert np.all(np.isfinite(features))


def test_extract_mfcc_accepts_short_signal() -> None:
    features = extract_mfcc(np.ones(100, dtype=np.float32) * 0.1)
    assert features.shape == (1, 26)


def test_command_features_add_pcen_speech_motion() -> None:
    samples = _tone(440)
    pcen = extract_pcen_cepstra(samples)
    command = extract_command_features(samples)

    assert command.shape[0] == pcen.shape[0]
    assert command.shape[1] == 26
    assert np.array_equal(command[:, :13], pcen)
    assert np.all(np.isfinite(command))


def test_pcen_accepts_short_signal() -> None:
    features = extract_pcen_cepstra(np.ones(100, dtype=np.float32) * 0.1)

    assert features.shape == (1, 13)
    assert np.all(np.isfinite(features))


def test_pcen_validates_smoothing() -> None:
    with np.testing.assert_raises(ValueError):
        extract_pcen_cepstra(_tone(440), smoothing=0.0)
