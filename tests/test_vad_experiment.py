import numpy as np

from ha_voice.vad_experiment import replay, speech_stream


def test_replay_silence_has_no_segments():
    events, timings, cpu = replay(np.zeros(32000, np.float32), "energy", 0.006, 1.7)
    assert events == []
    assert len(timings) == 50
    assert cpu >= 0


def test_stream_augmentation_is_reproducible_and_bounded():
    samples = np.ones(1600, np.float32)
    first = speech_stream(samples, 0.25, 0.0045, 7)
    assert np.array_equal(first, speech_stream(samples, 0.25, 0.0045, 7))
    assert len(first) == 36800
    assert abs(first[19200:20800].mean() - 0.25) < 0.001
    assert np.max(np.abs(first)) <= 1
