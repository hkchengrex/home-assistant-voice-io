"""Conservative trimming and level normalization for response WAV assets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import wave

import numpy as np

@dataclass(frozen=True)
class ProcessingResult:
    samples: np.ndarray
    sample_rate: int
    channels: int
    input_seconds: float
    output_seconds: float
    trim_start_seconds: float
    trim_end_seconds: float
    gain_db: float
    speech_rms_dbfs: float
    output_peak_dbfs: float


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def trim_and_normalize(
    samples: np.ndarray,
    sample_rate: int,
    *,
    silence_dbfs: float = -38.0,
    target_speech_rms_dbfs: float = -18.0,
    peak_dbfs: float = -1.0,
    frame_ms: int = 10,
    padding_ms: int = 0,
) -> ProcessingResult:
    """Trim only leading silence and normalize gated speech RMS."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    source = np.asarray(samples, dtype=np.float32)
    was_mono = source.ndim == 1
    if was_mono:
        source = source[:, np.newaxis]
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("samples must contain one or more audio frames")

    frame_length = max(1, round(sample_rate * frame_ms / 1000))
    frame_count = int(np.ceil(source.shape[0] / frame_length))
    padded = np.pad(source, ((0, frame_count * frame_length - source.shape[0]), (0, 0)))
    frames = padded.reshape(frame_count, frame_length, source.shape[1])
    frame_rms = np.sqrt(np.mean(frames * frames, axis=(1, 2)) + 1e-12)
    active = np.flatnonzero(frame_rms >= 10.0 ** (silence_dbfs / 20.0))
    if active.size == 0:
        raise ValueError("WAV contains no audio above the silence threshold")

    active_mask = frame_rms >= 10.0 ** (silence_dbfs / 20.0)
    onset_window_frames = max(1, round(50 / frame_ms))
    sustained = np.convolve(
        active_mask.astype(np.int16),
        np.ones(onset_window_frames, dtype=np.int16),
        mode="valid",
    )
    onset_candidates = np.flatnonzero(sustained >= max(1, onset_window_frames - 1))
    speech_start_frame = int(onset_candidates[0]) if onset_candidates.size else int(active[0])
    speech_start = speech_start_frame * frame_length
    padding = round(sample_rate * padding_ms / 1000)
    output_start = max(0, speech_start - padding)
    # The ending is intentionally inviolate. Generated response lines may finish
    # quietly, and an energy gate cannot safely distinguish that from silence.
    output_end = source.shape[0]
    trimmed = source[output_start:output_end].copy()

    active_samples = frames[active]
    speech_rms = float(np.sqrt(np.mean(active_samples * active_samples) + 1e-12))
    target_rms = 10.0 ** (target_speech_rms_dbfs / 20.0)
    desired_gain = target_rms / speech_rms
    source_peak = float(np.max(np.abs(trimmed)))
    peak_limit = 10.0 ** (peak_dbfs / 20.0)
    peak_gain = peak_limit / max(source_peak, 1e-12)
    gain = min(desired_gain, peak_gain)
    normalized = np.clip(trimmed * gain, -peak_limit, peak_limit).astype(np.float32)

    output = normalized[:, 0] if was_mono else normalized
    output_peak = float(np.max(np.abs(normalized)))
    return ProcessingResult(
        samples=output,
        sample_rate=sample_rate,
        channels=source.shape[1],
        input_seconds=source.shape[0] / sample_rate,
        output_seconds=normalized.shape[0] / sample_rate,
        trim_start_seconds=output_start / sample_rate,
        trim_end_seconds=(source.shape[0] - output_end) / sample_rate,
        gain_db=_dbfs(gain),
        speech_rms_dbfs=_dbfs(speech_rms * gain),
        output_peak_dbfs=_dbfs(output_peak),
    )


def process_wav(path: Path, output_path: Path | None = None) -> ProcessingResult:
    """Process a PCM16 WAV while preserving sample rate, channels, and its ending."""
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        compression = wav_file.getcomptype()
        raw = wav_file.readframes(wav_file.getnframes())
    if sample_width != 2 or compression != "NONE":
        raise ValueError(f"{path} must be uncompressed 16-bit PCM WAV")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels)
    result = trim_and_normalize(samples, sample_rate)
    output = result.samples
    if output.ndim == 1:
        output = output[:, np.newaxis]
    pcm = np.round(np.clip(output, -1.0, 1.0) * 32767.0).astype("<i2")

    destination = output_path or path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.wav")
    try:
        with wave.open(str(temporary), "wb") as wav_file:
            wav_file.setnchannels(result.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(result.sample_rate)
            wav_file.writeframes(pcm.tobytes())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _natural_key(path: Path) -> list[str | int]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.name)
    ]


def publish_asset_library(
    directory: Path, *, prefixes: set[str]
) -> list[tuple[Path, ProcessingResult]]:
    """Publish categorized source WAVs as the flat groups used by the listener."""
    published: list[tuple[Path, ProcessingResult]] = []
    for source_dir in sorted(
        path
        for path in directory.iterdir()
        if path.is_dir() and path.name in prefixes
    ):
        sources = sorted(source_dir.glob("*.wav"), key=_natural_key)
        if not sources:
            continue
        prefix = source_dir.name
        destinations: set[Path] = set()
        for index, source in enumerate(sources, start=1):
            destination = directory / f"{prefix}_{index}.wav"
            result = process_wav(source, destination)
            published.append((destination, result))
            destinations.add(destination)
        for stale in directory.glob(f"{prefix}_*.wav"):
            if stale not in destinations:
                stale.unlink()
    return published


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim and normalize response WAVs")
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish each categorized subfolder as a flat response group",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="response subfolder to publish; repeat for multiple groups",
    )
    args = parser.parse_args()
    processed = (
        publish_asset_library(args.directory, prefixes=set(args.prefix))
        if args.publish
        else [(path, process_wav(path)) for path in sorted(args.directory.glob("*.wav"))]
    )
    for path, result in processed:
        print(
            f"{path.name}: {result.input_seconds:.2f}s -> "
            f"{result.output_seconds:.2f}s, gain={result.gain_db:+.1f}dB, "
            f"speech={result.speech_rms_dbfs:.1f}dBFS, "
            f"peak={result.output_peak_dbfs:.1f}dBFS"
        )


if __name__ == "__main__":
    main()
