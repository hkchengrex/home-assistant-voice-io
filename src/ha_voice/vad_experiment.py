"""Replay saved speech through energy and WebRTC segmentation without actions.

This is a detector feasibility screen, not a recognition accuracy benchmark.
Input clips can be training data and lack human speech-boundary annotations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from .audio import load_wav
from .continuous import VoiceSegmenter
from .vad import WebRtcDetector


RATE = 16000
BLOCK = 320
VARIANTS = (
    ("original", 1.0, 0.0),
    ("quarter_volume", 0.25, 0.0),
    ("sixteenth_volume", 0.0625, 0.0),
    ("original_with_white_noise", 1.0, 0.0045),
    ("quarter_volume_with_white_noise", 0.25, 0.0045),
)


def replay(samples: np.ndarray, backend: str, min_rms: float, noise_multiplier: float):
    detector = None if backend == "energy" else WebRtcDetector(RATE, int(backend[-1]))
    segmenter = VoiceSegmenter(
        sample_rate=RATE, min_rms=min_rms, noise_multiplier=noise_multiplier,
        calibration_ms=1000, speech_detector=detector,
    )
    samples = np.pad(samples, (0, (-len(samples)) % BLOCK))
    events = []
    timings = []
    cpu_started = time.process_time()
    for start in range(0, len(samples), BLOCK):
        began = time.perf_counter()
        completed = segmenter.process(samples[start:start + BLOCK])
        elapsed = (time.perf_counter() - began) * 1000
        # Exclude calibration frames from detector processing measurements.
        if start >= RATE:
            timings.append(elapsed)
        for utterance in completed:
            events.append({
                "emitted_at_seconds": (start + BLOCK) / RATE,
                "retained_seconds": len(utterance.samples) / RATE,
                "duration_limit": utterance.hit_duration_limit,
            })
    cpu_seconds = time.process_time() - cpu_started
    return events, timings, cpu_seconds


def speech_stream(samples: np.ndarray, gain: float, noise_rms: float, seed: int):
    # Same exact waveform for every backend. Silence/noise prelude calibrates
    # energy detection and gives WebRTC 200 ms of adaptation before the clip.
    stream = np.pad(samples * gain, (round(1.2 * RATE), RATE)).astype(np.float32)
    if noise_rms:
        rng = np.random.default_rng(seed)
        stream += rng.normal(0, noise_rms, len(stream)).astype(np.float32)
    return np.clip(stream, -1, 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--include-label", action="append", required=True,
                        help="Explicit directory of saved speech; repeat for each label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-rms", type=float, default=0.004)
    parser.add_argument("--noise-multiplier", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args(argv)
    if args.min_rms <= 0 or args.noise_multiplier <= 0:
        parser.error("Detection thresholds must be positive")
    paths = []
    for label in sorted(set(args.include_label)):
        if label in (".", "..") or "/" in label or "\\" in label:
            parser.error("Labels must be immediate directory names")
        selected = sorted((args.recordings / label).glob("*.wav"))
        if not selected:
            parser.error(f"No WAV files in label {label}")
        paths.extend(selected)
    backends = ["energy", "webrtc0", "webrtc1", "webrtc2", "webrtc3"]
    timings = defaultdict(list)
    cpu = defaultdict(float)
    durations = defaultdict(float)
    details = []
    inventory = []
    for index, path in enumerate(paths):
        samples = load_wav(path, RATE).samples
        if samples.size == 0:
            raise ValueError(f"Empty recording: {path}")
        clip = path.relative_to(args.recordings).as_posix()
        inventory.append({"clip": clip, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "seconds": len(samples) / RATE})
        for variant, gain, noise in VARIANTS:
            stream = speech_stream(samples, gain, noise, args.seed + index)
            for backend in backends:
                events, frame_ms, cpu_seconds = replay(
                    stream, backend, args.min_rms, args.noise_multiplier)
                # Prelude events are excluded from the speech emission proxy.
                relevant = [event for event in events if event["emitted_at_seconds"] > 1.2]
                details.append({"clip": clip, "variant": variant, "backend": backend,
                                "events": relevant})
                timings[backend].extend(frame_ms)
                cpu[backend] += cpu_seconds
                durations[backend] += len(stream) / RATE
    noise_results = []
    rng = np.random.default_rng(args.seed)
    seconds = 30
    axis = np.arange(seconds * RATE, dtype=np.float32) / RATE
    noise_cases = {
        "silence": np.zeros(len(axis), np.float32),
        "white_noise_rms_0.0045": rng.normal(0, 0.0045, len(axis)).astype(np.float32),
        "hum_120hz_peak_0.03": (0.03 * np.sin(2 * np.pi * 120 * axis)).astype(np.float32),
    }
    for name, stream in noise_cases.items():
        for backend in backends:
            events, _, _ = replay(stream, backend, args.min_rms, args.noise_multiplier)
            noise_results.append({"case": name, "backend": backend,
                                  "seconds": seconds, "segments": len(events)})
    summary = []
    for variant, _, _ in VARIANTS:
        for backend in backends:
            rows = [row for row in details if row["variant"] == variant and row["backend"] == backend]
            summary.append({
                "variant": variant, "backend": backend, "clips": len(rows),
                "clips_with_segment": sum(bool(row["events"]) for row in rows),
                "clips_with_multiple_segments": sum(len(row["events"]) > 1 for row in rows),
                "clips_with_duration_limit": sum(any(e["duration_limit"] for e in row["events"]) for row in rows),
            })
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "Detector feasibility only; segment emission is not speech or command recall",
        "limitations": [
            "Saved clips may be training examples and may already be trimmed",
            "A segment can be noise, truncated speech, or the wrong boundaries",
            "White noise and gain scaling do not reproduce real room acoustics",
            "No recognition, wake gate, acknowledgement, or HA actions were run",
            "No human speech-end timestamps; end-to-end latency is not measured",
            "Thirty-second artificial noise cases cannot establish false activations per hour",
            "Existing energy-based final trimming is retained to isolate detector choice",
        ],
        "runtime": {"platform": platform.platform(), "python": platform.python_version(),
                    "numpy": np.__version__, "webrtcvad_wheels": version("webrtcvad-wheels")},
        "settings": {"sample_rate": RATE, "block_ms": 20, "start_ms": 80,
                     "end_silence_ms": 300, "pre_roll_ms": 700, "max_utterance_seconds": 2.5,
                     "min_rms": args.min_rms, "noise_multiplier": args.noise_multiplier,
                     "seed": args.seed},
        "inventory": inventory, "summary": summary, "noise_only": noise_results,
        "processing": {backend: {
            "frame_ms_median": float(np.median(timings[backend])),
            "frame_ms_p95": float(np.percentile(timings[backend], 95)),
            "cpu_seconds_per_audio_second": cpu[backend] / durations[backend],
        } for backend in backends},
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"clips": len(paths), "summary": summary,
                      "noise_only": noise_results, "processing": report["processing"]}, indent=2))


if __name__ == "__main__":
    main()
