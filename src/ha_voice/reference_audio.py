"""Local, bounded conversion of uploaded voice-cloning references.

PyAV is optional: recognition and playback do not import the media decoder.
"""

from fractions import Fraction
from io import BytesIO
import time
import wave

import numpy as np

REFERENCE_SAMPLE_RATE = 24_000
MAX_REFERENCE_BYTES = 10 * 1024 * 1024
_FORMATS = "wav,mp3,mov,flac,ogg,matroska,webm,aac,aiff"


def _reject_external_io(url, flags, options):
    raise ValueError("Reference audio must be a self-contained file")


def normalize_reference_audio(body: bytes) -> bytes:
    """Decode audio and return mono 24 kHz PCM16 WAV, without trimming speech.

    Only the supplied bytes are read. Playlists, external media references, and
    network access are disabled. Limits apply to decoded audio as well as upload
    size so a highly compressed long recording cannot fill memory.
    """
    if not 0 < len(body) <= MAX_REFERENCE_BYTES:
        raise ValueError("Choose an audio file of at most 10 MB")
    if body[:4] == b"RIFF" and body[8:12] == b"WAVE":
        # FFmpeg can salvage truncated WAVs; a voice reference must be complete.
        end = int.from_bytes(body[4:8], "little") + 8
        if end > len(body):
            raise ValueError("Use a complete reference recording, not a truncated WAV")
        position = 12
        while position + 8 <= end:
            size = int.from_bytes(body[position + 4:position + 8], "little")
            position += 8 + size
            if position > end:
                raise ValueError("Use a complete reference recording, not a truncated WAV")
            position += size % 2
    try:
        import av
    except ImportError as exc:
        raise ValueError('Audio conversion requires the voice-clone extra: pip install "voice-io[voice-clone]"') from exc

    started = time.monotonic()
    pcm = bytearray()
    seconds = Fraction(0)

    def append(frames):
        for frame in frames:
            if len(pcm) + frame.samples * 2 > 20 * REFERENCE_SAMPLE_RATE * 2:
                raise ValueError("Use a reference recording of 1–20 seconds")
            pcm.extend(frame.to_ndarray().astype("<i2", copy=False).tobytes())

    try:
        with av.open(BytesIO(body), mode="r", io_open=_reject_external_io,
                     timeout=10, options={"format_whitelist": _FORMATS,
                                          "protocol_whitelist": "none",
                                          "err_detect": "explode"}) as container:
            if len(container.streams.audio) != 1:
                raise ValueError("Choose a file with one audio track")
            stream = container.streams.audio[0]
            stream.codec_context.thread_count = 1
            resampler = av.AudioResampler(format="s16", layout="mono", rate=REFERENCE_SAMPLE_RATE)
            for packet in container.demux(stream):
                if time.monotonic() - started > 10:
                    raise ValueError("Audio conversion took too long; use a shorter reference")
                if packet.is_corrupt:
                    raise ValueError("Use a complete, undamaged reference recording")
                for frame in packet.decode():
                    if not 8000 <= frame.sample_rate <= 384000 or not 1 <= len(frame.layout.channels) <= 8:
                        raise ValueError("Reference audio must have 1–8 channels and an 8–384 kHz sample rate")
                    seconds += Fraction(frame.samples, frame.sample_rate)
                    if seconds > 20:
                        raise ValueError("Use a reference recording of 1–20 seconds")
                    if not np.isfinite(frame.to_ndarray()).all():
                        raise ValueError("Reference audio contains invalid sample values")
                    # Ignore container timestamp offsets; preserve decoded samples.
                    frame.pts = None
                    append(resampler.resample(frame))
            append(resampler.resample(None))
    except av.FFmpegError as exc:
        raise ValueError("Could not read this audio file. Use WAV, MP3, M4A/AAC, FLAC, OGG, WebM, or AIFF.") from exc

    if not REFERENCE_SAMPLE_RATE * 2 <= len(pcm) <= 20 * REFERENCE_SAMPLE_RATE * 2:
        raise ValueError("Use a reference recording of 1–20 seconds")
    result = BytesIO()
    with wave.open(result, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(REFERENCE_SAMPLE_RATE)
        output.writeframes(pcm)
    return result.getvalue()
