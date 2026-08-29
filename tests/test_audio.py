from io import BytesIO
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from ha_voice.audio import (
    Audio,
    decode_wav,
    encode_wav,
    record_audio,
    resample_linear,
    trim_silence,
    trim_spoken_phrase,
)


def test_record_audio_can_preserve_the_full_capture(monkeypatch) -> None:
    capture = np.linspace(-0.1, 0.1, 200, dtype=np.float32)[:, None]

    class CallbackStop(Exception):
        pass

    class FakeInputStream:
        def __init__(self, *, callback, **kwargs) -> None:
            self.callback = callback

        def __enter__(self):
            try:
                self.callback(capture, capture.shape[0], None, None)
            except CallbackStop:
                pass
            return self

        def __exit__(self, *args) -> None:
            return None

    fake_sounddevice = SimpleNamespace(
        CallbackStop=CallbackStop,
        InputStream=FakeInputStream,
        query_devices=lambda *args: {"default_samplerate": 200},
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    audio = record_audio(
        seconds=1.0,
        sample_rate=100,
        countdown=0,
        trim=False,
    )

    assert audio.samples.shape == (100,)
    assert np.isclose(audio.samples[0], capture[0, 0])
    assert np.isclose(audio.samples[-1], capture[-1, 0])


def test_resample_linear_changes_length() -> None:
    source = np.linspace(-1, 1, 8000, dtype=np.float32)
    result = resample_linear(source, 8000, 16000)
    assert result.shape == (16000,)
    assert np.isclose(result[0], -1)
    assert np.isclose(result[-1], 1)


def test_trim_silence_keeps_signal_and_padding() -> None:
    sample_rate = 16000
    signal = np.concatenate(
        (
            np.zeros(sample_rate, dtype=np.float32),
            np.full(sample_rate // 2, 0.4, dtype=np.float32),
            np.zeros(sample_rate, dtype=np.float32),
        )
    )
    trimmed = trim_silence(signal, sample_rate, padding_ms=100)
    assert sample_rate // 2 < trimmed.size < sample_rate
    assert np.max(trimmed) == np.float32(0.4)


def test_trim_silence_ignores_distant_noise_spike() -> None:
    sample_rate = 16000
    signal = np.zeros(sample_rate * 3, dtype=np.float32)
    signal[200:520] = 0.08
    signal[sample_rate : sample_rate + 8000] = 0.3

    trimmed = trim_silence(signal, sample_rate)

    assert 0.5 < trimmed.size / sample_rate < 1.0
    assert np.max(trimmed) == np.float32(0.3)


def test_trim_spoken_phrase_keeps_words_around_natural_pause() -> None:
    sample_rate = 16000
    signal = np.concatenate(
        (
            np.zeros(sample_rate // 2, dtype=np.float32),
            np.full(sample_rate // 4, 0.15, dtype=np.float32),
            np.zeros(sample_rate // 2, dtype=np.float32),
            np.full(sample_rate // 3, 0.3, dtype=np.float32),
            np.zeros(sample_rate // 2, dtype=np.float32),
        )
    )

    strict = trim_silence(signal, sample_rate)
    phrase = trim_spoken_phrase(signal, sample_rate)

    assert phrase.size > strict.size + sample_rate // 2
    assert 1.0 < phrase.size / sample_rate < 1.8


def test_decode_wav_downmixes_and_resamples() -> None:
    samples = np.tile(np.array([0, 1000], dtype="<i2"), (8000, 1))
    encoded = BytesIO()
    with wave.open(encoded, "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(samples.tobytes())
    audio = decode_wav(encoded.getvalue(), 16000)
    assert audio.sample_rate == 16000
    assert audio.samples.shape == (16000,)
    assert np.isclose(audio.samples.mean(), 500 / 32768, atol=1e-4)


def test_decode_wav_rejects_malformed_audio() -> None:
    with pytest.raises(ValueError, match="valid PCM WAV"):
        decode_wav(b"not a wave file")


def test_encode_wav_round_trips_mono_audio() -> None:
    source = np.linspace(-0.5, 0.5, 1600, dtype=np.float32)
    decoded = decode_wav(encode_wav(Audio(source, 16000)))
    assert decoded.sample_rate == 16000
    assert decoded.samples.shape == source.shape
    assert np.allclose(decoded.samples, source, atol=2 / 32767)
