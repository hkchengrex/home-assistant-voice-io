from io import BytesIO
import sys
import wave

import av
import numpy as np
import pytest

from ha_voice.reference_audio import MAX_REFERENCE_BYTES, normalize_reference_audio


def encoded_audio(container_format="wav", codec="pcm_s16le", rate=48000,
                  layout="stereo", seconds=2, frequency=180):
    """Synthesize fixtures, including codec headers and end-of-stream flushing."""
    samples = (.1 * np.sin(np.arange(round(rate * seconds)) * 2 * np.pi * frequency / rate)).astype(np.float32)
    channels = len(av.AudioLayout(layout).channels)
    output = BytesIO()
    with av.open(output, "w", format=container_format) as container:
        stream = container.add_stream(codec, rate=rate)
        stream.layout = layout
        if codec == "vorbis":
            stream.options = {"strict": "experimental"}
        for start in range(0, len(samples), 4096):
            frame = av.AudioFrame.from_ndarray(np.tile(samples[start:start + 4096], (channels, 1)),
                                              format="fltp", layout=layout)
            frame.sample_rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return output.getvalue()


def decoded_reference(body):
    with wave.open(BytesIO(body), "rb") as output:
        assert output.getframerate() == 24000
        assert output.getnchannels() == 1
        assert output.getsampwidth() == 2
        return np.frombuffer(output.readframes(output.getnframes()), dtype="<i2").astype(float) / 32768


@pytest.mark.parametrize("codec", ["pcm_u8", "pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_f32le", "pcm_f64le"])
@pytest.mark.parametrize("rate", [8000, 44100, 48000, 96000, 192000])
def test_wav_bit_depth_and_sample_rate_conversion(codec, rate):
    audio = decoded_reference(normalize_reference_audio(encoded_audio(codec=codec, rate=rate)))
    assert len(audio) == 48000
    frequency = np.argmax(np.abs(np.fft.rfft(audio))) * 24000 / len(audio)
    assert frequency == pytest.approx(180, abs=1)
    assert .04 < np.sqrt(np.mean(audio ** 2)) < .12


@pytest.mark.parametrize("container_format,codec", [
    ("mp3", "libmp3lame"), ("mp4", "aac"), ("adts", "aac"),
    ("flac", "flac"), ("ogg", "vorbis"), ("ogg", "libopus"),
    ("webm", "libopus"), ("aiff", "pcm_s24be"),
])
def test_compressed_and_other_containers(container_format, codec):
    audio = decoded_reference(normalize_reference_audio(encoded_audio(container_format, codec)))
    assert len(audio) / 24000 == pytest.approx(2, abs=.08)  # Lossy codec padding.
    assert .04 < np.sqrt(np.mean(audio ** 2)) < .12


def test_multichannel_downmix_and_resampler_antialiasing():
    audio = decoded_reference(normalize_reference_audio(encoded_audio(layout="5.1")))
    assert len(audio) == 48000 and np.sqrt(np.mean(audio ** 2)) > .01
    high = decoded_reference(normalize_reference_audio(encoded_audio(frequency=18000)))
    assert np.sqrt(np.mean(high[100:-100] ** 2)) < .001


@pytest.mark.parametrize("seconds", [.5, 21])
def test_duration_limit_applies_to_decoded_compressed_audio(seconds):
    with pytest.raises(ValueError, match="1–20 seconds"):
        normalize_reference_audio(encoded_audio("flac", "flac", seconds=seconds))


@pytest.mark.parametrize("seconds", [1, 20])
def test_exact_duration_boundaries(seconds):
    audio = decoded_reference(normalize_reference_audio(encoded_audio(rate=44100, seconds=seconds)))
    assert len(audio) == seconds * 24000


@pytest.mark.parametrize("body", [b"", b"x" * (MAX_REFERENCE_BYTES + 1), b"not audio",
    b"#EXTM3U\n#EXTINF:2,\nhttp://127.0.0.1:9/private.wav\n",
    b"ffconcat version 1.0\nfile '/etc/passwd'\n"], ids=["empty", "oversized", "invalid", "playlist", "concat"])
def test_invalid_oversized_and_playlist_uploads(body):
    with pytest.raises(ValueError):
        normalize_reference_audio(body)


def test_truncated_wav_is_not_silently_salvaged():
    with pytest.raises(ValueError, match="complete"):
        normalize_reference_audio(encoded_audio()[:-100])


def test_missing_optional_dependency_is_actionable(monkeypatch):
    body = encoded_audio()
    monkeypatch.setitem(sys.modules, "av", None)
    with pytest.raises(ValueError, match="voice-clone"):
        normalize_reference_audio(body)


def test_conversion_is_idempotent_for_canonical_wav():
    body = normalize_reference_audio(encoded_audio())
    assert normalize_reference_audio(body) == body
