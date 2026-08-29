import numpy as np

from ha_voice.continuous import VoiceSegmenter


def _blocks(value: float, count: int, block_size: int = 320) -> list[np.ndarray]:
    return [np.full(block_size, value, dtype=np.float32) for _ in range(count)]


def test_segmenter_emits_after_trailing_pause() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    stream = _blocks(0.001, 10) + _blocks(0.08, 30) + _blocks(0.001, 15)
    for block in stream:
        output.extend(segmenter.process(block))

    assert len(output) == 1
    assert 0.5 < output[0].samples.size / segmenter.sample_rate <= 1.0
    assert output[0].hit_duration_limit is False
    assert not segmenter.active


def test_segmenter_ignores_single_noise_spike() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    stream = _blocks(0.001, 10) + _blocks(0.08, 1) + _blocks(0.001, 40)
    for block in stream:
        output.extend(segmenter.process(block))

    assert output == []
    assert not segmenter.active


def test_segmenter_emits_multiple_commands() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    utterance = _blocks(0.08, 25) + _blocks(0.001, 30)
    for block in _blocks(0.001, 10) + utterance + utterance:
        output.extend(segmenter.process(block))

    assert len(output) == 2


def test_segmenter_calibrates_to_steady_room_noise() -> None:
    segmenter = VoiceSegmenter(
        min_rms=0.006,
        noise_multiplier=1.7,
        calibration_ms=1000,
    )
    output: list[np.ndarray] = []
    for block in _blocks(0.007, 100):
        output.extend(segmenter.process(block))

    assert output == []
    assert not segmenter.active
    assert segmenter.noise_rms > 0.006


def test_segmenter_hears_speech_after_room_calibration() -> None:
    segmenter = VoiceSegmenter(
        min_rms=0.006,
        noise_multiplier=1.7,
        calibration_ms=1000,
    )
    output: list[np.ndarray] = []
    stream = _blocks(0.007, 50) + _blocks(0.05, 30) + _blocks(0.001, 15)
    for block in stream:
        output.extend(segmenter.process(block))

    assert len(output) == 1


def test_segmenter_caps_noisy_utterance_at_two_and_a_half_seconds() -> None:
    segmenter = VoiceSegmenter()
    output: list[np.ndarray] = []
    for block in _blocks(0.08, 140):
        output.extend(segmenter.process(block))

    assert len(output) == 1
    assert output[0].samples.size / segmenter.sample_rate <= 2.5
    assert output[0].hit_duration_limit is True


def test_segmenter_pre_roll_recovers_quiet_first_word() -> None:
    segmenter = VoiceSegmenter(
        min_rms=0.006,
        noise_multiplier=1.7,
        pre_roll_ms=700,
    )
    output: list[np.ndarray] = []
    stream = (
        _blocks(0.001, 10)
        + _blocks(0.005, 15)
        + _blocks(0.001, 5)
        + _blocks(0.05, 10)
        + _blocks(0.001, 15)
    )
    for block in stream:
        output.extend(segmenter.process(block))

    assert len(output) == 1
    assert np.any(np.isclose(output[0].samples, 0.005))
