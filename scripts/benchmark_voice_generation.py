"""Compare separate requests with shared-prompt sequential and batched generation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import statistics
import time
import wave


TEXTS = [
    "Welcome home. It is good to see you.",
    "The lights are on. Have a pleasant evening.",
    "Everything is ready. You can relax now.",
    "Good night. Sleep well and see you tomorrow.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--confirm-upload", action="store_true")
    args = parser.parse_args()
    if not args.confirm_upload:
        parser.error("--confirm-upload is required to send reference audio to Hugging Face")
    if not args.reference.is_file() or args.repetitions < 1 or not 4 <= args.steps <= 64:
        parser.error("Use an existing reference, positive repetitions, and 4–64 steps")

    import httpx
    from gradio_client import Client, handle_file
    from huggingface_hub import HfApi, get_token, set_client_factory

    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(retries=3), timeout=30, follow_redirects=True
    ))
    client = Client("https://hkchengrex-omnivoice.hf.space", token=get_token(), verbose=False)
    upload = handle_file(str(args.reference.resolve()))
    controls = ("English", upload, args.reference_text, "", args.steps, 2.0, True, 1.0, None, True, True)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "space_revision": HfApi().space_info("hkchengrex/OmniVoice").sha,
        "steps": args.steps, "texts": TEXTS, "repetitions": args.repetitions,
        "warmup_requests": 1, "reference_transcript_supplied": True, "runs": [],
    }

    def save_report():
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Warming the single-response endpoint...", flush=True)
    warm = client.predict(TEXTS[0], *controls, api_name="/_clone_fn")
    if not str(warm[1]).startswith("Done"):
        raise RuntimeError(f"Warm-up failed: {warm[1]}")
    print("Warm-up complete (excluded from results).", flush=True)
    modes = ("single_requests", "batch_size_1", "batch_size_4")
    for repetition in range(args.repetitions):
        # Rotate order so every mode is not always measured in the same position.
        order = modes[repetition % 3:] + modes[:repetition % 3]
        for mode in order:
            print(f"Run {repetition + 1}: {mode}", flush=True)
            start = time.perf_counter()
            metrics = None
            try:
                if mode == "single_requests":
                    paths = []
                    for text in TEXTS:
                        path, status = client.predict(text, *controls, api_name="/_clone_fn")
                        if not str(status).startswith("Done"):
                            raise RuntimeError(status)
                        paths.append(path)
                else:
                    size = 1 if mode == "batch_size_1" else 4
                    paths, metrics = client.predict(TEXTS, *controls, size, api_name="/clone_batch")
                    if metrics.get("status") != "done" or metrics.get("count") != len(TEXTS):
                        raise RuntimeError("Incomplete batch")
                elapsed = time.perf_counter() - start
                durations = []
                for index, path in enumerate(paths):
                    with wave.open(str(path), "rb") as wav:
                        if wav.getsampwidth() != 2 or wav.getnframes() == 0:
                            raise RuntimeError("Invalid generated PCM audio")
                        durations.append(wav.getnframes() / wav.getframerate())
                    shutil.copyfile(path, args.output / f"{mode}-{repetition + 1}-{index + 1}.wav")
                row = {"mode": mode, "repetition": repetition + 1, "wall_seconds": elapsed,
                       "audio_seconds": durations, "metrics": metrics}
                report["runs"].append(row)
                save_report()
                print(json.dumps(row), flush=True)
            except Exception as exc:
                report["error"] = {"mode": mode, "repetition": repetition + 1,
                                   "type": type(exc).__name__, "detail": str(exc)[:500]}
                save_report()
                raise
    medians = {
        mode: statistics.median(row["wall_seconds"] for row in report["runs"] if row["mode"] == mode)
        for mode in modes
    }
    report["median_wall_seconds"] = medians
    report["wall_speedup"] = medians["single_requests"] / medians["batch_size_4"]
    save_report()
    print(json.dumps({"medians": medians, "wall_speedup": report["wall_speedup"]}), flush=True)


if __name__ == "__main__":
    main()
