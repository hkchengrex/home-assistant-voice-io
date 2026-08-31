"""Bounded, true model batching for the OmniVoice Space (Apache-2.0)."""

from __future__ import annotations

import math
import time

import numpy as np

MAX_ITEMS = 8
MAX_TEXT_CHARS = 300
MAX_TOTAL_CHARS = 1200
MAX_REFERENCE_SECONDS = 20


def validate_request(texts, batch_size, num_step, guidance_scale, speed, duration):
    """Reject unbounded jobs before acquiring a ZeroGPU allocation."""
    if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_ITEMS:
        raise ValueError(f"texts must be a JSON list of 1–{MAX_ITEMS} strings")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Each text must be a non-empty string")
    texts = [text.strip() for text in texts]
    if any(len(text) > MAX_TEXT_CHARS for text in texts):
        raise ValueError(f"Each text is limited to {MAX_TEXT_CHARS} characters")
    if sum(map(len, texts)) > MAX_TOTAL_CHARS:
        raise ValueError(f"A request is limited to {MAX_TOTAL_CHARS} characters")
    for name, value, low, high, integer in (
        ("batch_size", batch_size, 1, MAX_ITEMS, True),
        ("num_step", num_step, 4, 64, True),
        ("guidance_scale", guidance_scale, 0, 4, False),
        ("speed", speed, 0.5, 1.5, False),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not low <= value <= high
            or (integer and int(value) != value)
        ):
            raise ValueError(f"{name} must be {'an integer ' if integer else ''}between {low} and {high}")
    if duration is not None and (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not 0 < duration <= 15
    ):
        raise ValueError("duration must be positive and at most 15 seconds per clip")
    return texts


def generate_batch(
    model, config_factory, texts, language, reference_audio, reference_text,
    instruct, num_step, guidance_scale, denoise, speed, duration,
    preprocess_prompt, postprocess_output, batch_size=4,
    *, synchronize=lambda: None, device_name="unknown",
):
    """Prepare one prompt, then generate lists of texts in bounded microbatches."""
    texts = validate_request(texts, batch_size, num_step, guidance_scale, speed, duration)
    synchronize()
    start = time.perf_counter()
    prompt = model.create_voice_clone_prompt(
        ref_audio=reference_audio,
        ref_text=reference_text.strip() or None,
        preprocess_prompt=preprocess_prompt,
    )
    synchronize()
    prompt_seconds = time.perf_counter() - start
    config = config_factory(
        num_step=int(num_step), guidance_scale=float(guidance_scale),
        denoise=denoise, preprocess_prompt=preprocess_prompt,
        postprocess_output=postprocess_output,
    )
    generated = []
    generation_start = time.perf_counter()
    for offset in range(0, len(texts), int(batch_size)):
        chunk = texts[offset:offset + int(batch_size)]
        audio = model.generate(
            text=chunk,
            language=None if language in (None, "", "Auto") else language,
            voice_clone_prompt=prompt,
            instruct=instruct.strip() or None,
            speed=float(speed), duration=duration, generation_config=config,
        )
        if len(audio) != len(chunk):
            raise RuntimeError("The model returned an incomplete batch")
        for waveform in audio:
            waveform = np.asarray(waveform)
            if waveform.ndim != 1 or not waveform.size or not np.isfinite(waveform).all():
                raise RuntimeError("The model returned invalid audio")
            generated.append(np.round(np.clip(waveform, -1, 1) * 32767).astype("<i2"))
    synchronize()
    end = time.perf_counter()
    return generated, {
        "schema_version": 1, "status": "done", "count": len(generated),
        "batch_size": int(batch_size), "sample_rate": model.sampling_rate,
        "gpu": device_name, "prompt_seconds": prompt_seconds,
        "generation_seconds": end - generation_start,
        "gpu_task_seconds": end - start,
        "items": [
            {"index": index, "audio_seconds": len(audio) / model.sampling_rate}
            for index, audio in enumerate(generated)
        ],
    }


def attach_batch_api(demo, model, gpu_generate):
    """Add /clone_batch without replacing the upstream single-clip UI or API."""
    import io
    import wave

    import gradio as gr
    import soundfile as sf
    from gradio.processing_utils import save_bytes_to_cache

    for function in demo.fns.values():
        function.concurrency_id = "omnivoice-gpu"
        function.concurrency_limit = 1
    demo.delete_cache = (3600, 3600)

    with demo:
        with gr.Accordion("Batch voice cloning API", open=False):
            gr.Markdown("Generate up to 8 short lines using one reference. Outputs follow input order.")
            texts = gr.JSON(value=["Welcome home.", "Good night."], label="Texts (JSON list)")
            language = gr.Textbox(value="Auto", label="Language")
            reference = gr.Audio(type="filepath", label="Reference audio (1–20 seconds)")
            transcript = gr.Textbox(value="", label="Reference transcript (optional)")
            instruct = gr.Textbox(value="", label="Voice direction (optional)")
            steps = gr.Slider(4, 64, value=32, step=1, label="Inference steps")
            guidance = gr.Slider(0, 4, value=2.0, label="Guidance scale")
            denoise = gr.Checkbox(value=True, label="Denoise")
            speed = gr.Slider(0.5, 1.5, value=1.0, label="Speed")
            duration = gr.Number(value=None, label="Duration per clip (optional, up to 15s)")
            preprocess = gr.Checkbox(value=True, label="Preprocess reference")
            postprocess = gr.Checkbox(value=True, label="Postprocess output")
            size = gr.Slider(1, MAX_ITEMS, value=4, step=1, label="GPU batch size")
            generate = gr.Button("Generate batch")
            files = gr.File(file_count="multiple", label="Generated WAV files")
            report = gr.JSON(label="Timing report (excludes queue and download)")

        def clone_batch(texts, language, reference_audio, reference_text, instruct,
                        num_step, guidance_scale, denoise, speed, duration,
                        preprocess_prompt, postprocess_output, batch_size):
            try:
                texts = validate_request(texts, batch_size, num_step, guidance_scale, speed, duration)
                if not reference_audio:
                    raise ValueError("Upload a reference audio file")
                info = sf.info(reference_audio)
                if not 1 <= info.duration <= MAX_REFERENCE_SECONDS:
                    raise ValueError("Reference audio must be between 1 and 20 seconds")
                if info.channels > 2 or info.samplerate > 192000:
                    raise ValueError("Use mono or stereo reference audio at up to 192 kHz")
                audios, metrics = gpu_generate(
                    texts, language, reference_audio, reference_text or "", instruct or "",
                    num_step, guidance_scale, denoise, speed, duration,
                    preprocess_prompt, postprocess_output, batch_size,
                )
            except ValueError as exc:
                raise gr.Error(str(exc)) from exc
            paths = []
            for index, audio in enumerate(audios):
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(model.sampling_rate)
                    wav.writeframes(audio.tobytes())
                path = save_bytes_to_cache(buffer.getvalue(), f"response_{index + 1:04d}.wav", files.GRADIO_CACHE)
                files.temp_files.add(path)
                paths.append(path)
            return paths, metrics

        generate.click(
            clone_batch,
            inputs=[texts, language, reference, transcript, instruct, steps, guidance,
                    denoise, speed, duration, preprocess, postprocess, size],
            outputs=[files, report], api_name="clone_batch",
            concurrency_id="omnivoice-gpu", concurrency_limit=1,
        )
    return demo
