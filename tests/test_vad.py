import numpy as np
import pytest

from ha_voice.continuous import ContinuousListener, VoiceSegmenter
from ha_voice.vad import WebRtcDetector


def test_speech_detector_can_recover_audio_below_energy_floor():
    segmenter = VoiceSegmenter(
        min_rms=0.1, speech_detector=lambda block: bool(np.max(block) > 0.0005)
    )
    frames = [np.zeros(320, np.float32)] * 50
    frames += [np.full(320, 0.001, np.float32)] * 30
    frames += [np.zeros(320, np.float32)] * 20
    output = [item for frame in frames for item in segmenter.process(frame)]
    assert len(output) == 1
    assert np.max(output[0].samples) == pytest.approx(0.001)


def test_detector_rejects_loud_non_speech_without_energy_override():
    segmenter = VoiceSegmenter(speech_detector=lambda block: False)
    assert all(not segmenter.process(np.full(320, 0.5)) for _ in range(200))


def test_invalid_mode_and_rate_fail_before_import():
    with pytest.raises(ValueError):
        WebRtcDetector(mode=4)
    with pytest.raises(ValueError):
        WebRtcDetector(sample_rate=22050)


def test_webrtc_silence_and_input_validation():
    pytest.importorskip("webrtcvad")
    detector = WebRtcDetector()
    assert not detector(np.zeros(320, np.float32))
    for samples in (np.zeros(319), np.zeros((320, 1)), np.full(320, np.nan)):
        with pytest.raises(ValueError):
            detector(samples)


def test_listener_keeps_detector_choice_after_feedback_reset():
    pytest.importorskip("webrtcvad")
    listener = ContinuousListener(vad_backend="webrtc", vad_mode=1)
    first = listener._new_segmenter(calibration_ms=1000)
    replacement = listener._new_segmenter()
    assert isinstance(first.speech_detector, WebRtcDetector)
    assert isinstance(replacement.speech_detector, WebRtcDetector)
    assert first.speech_detector is not replacement.speech_detector
    assert not replacement.speech_detector(np.zeros(320))


def test_listener_default_and_invalid_detector_selection():
    assert ContinuousListener()._new_segmenter().speech_detector is None
    with pytest.raises(ValueError):
        ContinuousListener(vad_backend="unknown")
    with pytest.raises(ValueError):
        ContinuousListener(vad_mode=4)
