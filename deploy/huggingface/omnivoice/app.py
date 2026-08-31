#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Derived from hkchengrex/OmniVoice, revision 64deef91cf1f7bfdfc97ff07502b4cd47a0e361e.
"""
HuggingFace Space entry point for OmniVoice demo.

"""

import logging
import os
from typing import Any, Dict

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

import numpy as np
import spaces
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.cli.demo import build_demo
from batch_api import attach_batch_api, generate_batch

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
CHECKPOINT = os.environ.get("OMNIVOICE_MODEL", "k2-fsa/OmniVoice")

print(f"Loading model from {CHECKPOINT} to cuda ...")
model = OmniVoice.from_pretrained(
    CHECKPOINT,
    device_map="cuda",
    dtype=torch.float16,
    load_asr=True,
)
sampling_rate = model.sampling_rate
print("Model loaded successfully!")

# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def _gen_core(
    text,
    language,
    ref_audio,
    instruct,
    num_step,
    guidance_scale,
    denoise,
    speed,
    duration,
    preprocess_prompt,
    postprocess_output,
    mode,
    ref_text=None,
):
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    )

    lang = language if (language and language != "Auto") else None

    kw: Dict[str, Any] = dict(
        text=text.strip(), language=lang, generation_config=gen_config
    )

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
        )

    if instruct and instruct.strip():
        kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    waveform = (audio[0] * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."


# ---------------------------------------------------------------------------
# ZeroGPU wrapper
# ---------------------------------------------------------------------------


@spaces.GPU(duration=60)
def generate_fn(*args, **kwargs):
    return _gen_core(*args, **kwargs)


@spaces.GPU(duration=120)
def generate_batch_fn(*args):
    return generate_batch(
        model, OmniVoiceGenerationConfig, *args,
        synchronize=torch.cuda.synchronize,
        device_name=torch.cuda.get_device_name(),
    )


# ---------------------------------------------------------------------------
# Build and launch demo
# ---------------------------------------------------------------------------
demo = build_demo(model, CHECKPOINT, generate_fn=generate_fn)
demo = attach_batch_api(demo, model, generate_batch_fn)

if __name__ == "__main__":
    demo.queue().launch()
