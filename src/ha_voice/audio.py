"""Audio capture and WAV helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import threading
import time
import wave

import numpy as np


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int


def _read_wav(source: str | BytesIO, target_sample_rate: int) -> Audio:
    with wave.open(source, "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width != 2:
        raise ValueError("Audio must be 16-bit PCM WAV")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if source_rate != target_sample_rate:
        samples = resample_linear(samples, source_rate, target_sample_rate)
    return Audio(samples=samples, sample_rate=target_sample_rate)


def load_wav(path: Path, target_sample_rate: int = 16_000) -> Audio:
    return _read_wav(str(path), target_sample_rate)


def decode_wav(data: bytes, target_sample_rate: int = 16_000) -> Audio:
    try:
        return _read_wav(BytesIO(data), target_sample_rate)
    except (EOFError, wave.Error) as exc:
        raise ValueError("Audio must be a valid PCM WAV recording") from exc


def save_wav(path: Path, audio: Audio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_wav(audio))


def encode_wav(audio: Audio) -> bytes:
    samples = np.clip(audio.samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2")
    encoded = BytesIO()
    with wave.open(encoded, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(audio.sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return encoded.getvalue()


def resample_linear(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if samples.size == 0 or source_rate == target_rate:
        return samples.astype(np.float32, copy=True)
    target_length = max(1, round(samples.size * target_rate / source_rate))
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.linspace(0, samples.size - 1, target_length)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    relative_db: float = -24.0,
    padding_ms: int = 120,
    noise_multiplier: float = 2.8,
    max_gap_ms: int = 300,
) -> np.ndarray:
    """Keep the strongest contiguous speech region and a small edge padding."""
    if samples.size == 0:
        return samples.copy()
    frame_length = max(1, round(sample_rate * frame_ms / 1000))
    frame_count = int(np.ceil(samples.size / frame_length))
    padded = np.pad(samples, (0, frame_count * frame_length - samples.size))
    frames = padded.reshape(frame_count, frame_length)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    peak_rms = float(rms.max())
    noise_floor = float(np.percentile(rms, 20))
    adaptive_noise_threshold = min(noise_floor * noise_multiplier, peak_rms * 0.5)
    threshold = max(
        peak_rms * (10.0 ** (relative_db / 20.0)),
        adaptive_noise_threshold,
        1e-4,
    )
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return np.empty(0, dtype=np.float32)

    # A single noise spike near either edge should not stretch a short command
    # across the entire capture. Bridge normal between-word pauses, split larger
    # gaps, then keep the group containing the most acoustic energy.
    max_gap_frames = max(1, round(max_gap_ms / frame_ms))
    groups: list[tuple[int, int]] = []
    group_start = previous = int(active[0])
    for frame_index in active[1:]:
        frame_index = int(frame_index)
        if frame_index - previous > max_gap_frames:
            groups.append((group_start, previous))
            group_start = frame_index
        previous = frame_index
    groups.append((group_start, previous))
    first_active, last_active = max(
        groups,
        key=lambda group: float(np.sum(rms[group[0] : group[1] + 1] ** 2)),
    )

    padding = round(sample_rate * padding_ms / 1000)
    start = max(0, first_active * frame_length - padding)
    end = min(samples.size, (last_active + 1) * frame_length + padding)
    return samples[start:end].astype(np.float32, copy=True)


def trim_spoken_phrase(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim a short phrase without dropping words around a natural pause."""
    return trim_silence(
        samples,
        sample_rate,
        padding_ms=200,
        max_gap_ms=800,
    )


def list_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install the sounddevice package to access microphones") from exc
    return [dict(device) for device in sd.query_devices()]


def record_audio(
    seconds: float,
    sample_rate: int = 16_000,
    device: int | str | None = None,
    countdown: int = 2,
    *,
    trim: bool = True,
) -> Audio:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install the sounddevice package to record audio") from exc

    if seconds <= 0:
        raise ValueError("seconds must be positive")
    for remaining in range(countdown, 0, -1):
        print(f"Recording in {remaining}...")
        time.sleep(1)
    try:
        device_info = sd.query_devices(device, "input")
        source_rate = round(float(device_info.get("default_samplerate", sample_rate)))
    except Exception:  # Query behavior varies across PortAudio host APIs.
        source_rate = sample_rate
    if source_rate <= 0:
        source_rate = sample_rate

    print("Speak now.")
    frame_target = max(1, round(seconds * source_rate))
    captured: list[np.ndarray] = []
    finished = threading.Event()
    remaining = frame_target

    def callback(
        input_data: np.ndarray,
        frames: int,
        timing: object,
        status: object,
    ) -> None:
        del timing, status
        nonlocal remaining
        take = min(frames, remaining)
        captured.append(input_data[:take, 0].copy())
        remaining -= take
        if remaining <= 0:
            finished.set()
            raise sd.CallbackStop

    try:
        with sd.InputStream(
            samplerate=source_rate,
            channels=1,
            dtype="float32",
            device=device,
            callback=callback,
        ):
            if not finished.wait(seconds + 2.0):
                raise RuntimeError("The microphone opened but did not return audio")
    except RuntimeError:
        raise
    except Exception as exc:  # PortAudio exposes backend-specific exception types.
        raise RuntimeError(f"Could not record from this microphone: {exc}") from exc

    if not captured:
        raise RuntimeError("The microphone returned no audio")
    samples = np.concatenate(captured)[:frame_target]
    if source_rate != sample_rate:
        samples = resample_linear(samples, source_rate, sample_rate)
    print("Captured.")
    if not trim:
        return Audio(samples=samples, sample_rate=sample_rate)
    samples = trim_spoken_phrase(samples, sample_rate)
    if samples.size < sample_rate // 5:
        raise RuntimeError("Capture contained too little non-silent audio")
    return Audio(samples=samples, sample_rate=sample_rate)
