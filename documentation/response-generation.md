# Generate response audio

Home Assistant Voice IO can use the [OmniVoice Hugging Face Space](https://huggingface.co/spaces/hkchengrex/OmniVoice) to create spoken response clips from a reference voice. The generated WAV files are normalized and published directly into the response library used by the listener.

!!! warning "Reference audio is uploaded"
    Recognition stays local, but this optional generation workflow sends the selected reference recording and response text to the Hugging Face Space. Use a voice only with the speaker's permission. The command requires `--confirm-upload` so this cannot happen accidentally.

## Install the optional client

=== "From a checkout"

    ```shell
    python -m pip install -e ".[capture,voice-clone]"
    ```

=== "From a package release"

    After a `voice-io` release is published:

    ```shell
    python -m pip install "voice-io[capture,voice-clone]"
    ```

## Prepare the reference

Use a clean recording of roughly 3–10 seconds with one speaker, little background noise, and no music. A transcript of the reference is optional because the Space can transcribe it, but supplying the exact words usually removes ambiguity.

## Generate one response

`--response` is a name from the `[responses]` section of `commands.toml`:

```shell
voice-io --config voice-data/commands.toml clone-response \
  --response welcome \
  --text "Welcome home" \
  --reference voice-data/reference.wav \
  --reference-text "This is the voice reference." \
  --language English \
  --assets voice-data/assets \
  --confirm-upload
```

On PowerShell, put the command on one line or replace each `\` continuation with a backtick.

The command saves two forms:

```text
voice-data/assets/
├── welcome/
│   └── omnivoice_0001.wav   # private generated source
└── welcome_1.wav            # normalized clip used for playback
```

Generate another line with the same response name to add variety. The response player chooses between the published clips without immediately repeating one.

## Generate a batch

Put 1–8 lines in a UTF-8 file such as `welcome-lines.txt`, one clip per non-empty
line. All lines share the reference voice, language, and generation settings:

```text
Welcome home.
Good to see you again.
I hope you had a pleasant day.
Make yourself comfortable.
```

```shell
voice-io --config voice-data/commands.toml clone-response \
  --response welcome \
  --text-file welcome-lines.txt \
  --reference voice-data/reference.wav \
  --reference-text "This is the voice reference." \
  --language English \
  --batch-size 4 \
  --assets voice-data/assets \
  --confirm-upload
```

The Space processes the reference once, then generates multiple lines together
on the GPU. `--batch-size` controls how many lines are generated at once; the
default is 4 and the maximum is 8. Output files follow input order and are added
to the chosen response group. Existing source clips are preserved. The client
validates and normalizes the whole batch before updating the library; do not
run multiple publishers against the same library simultaneously.

Batch requests are limited to 300 characters per line, 1200 total characters,
and a 1–20 second reference. A supplied `--duration` may be at most 15 seconds
per clip. For longer material, split the request. On GPU memory errors, reduce
the batch size. The client does not silently switch to separate requests.

The existing `--text` command still uses the original single-response endpoint.
Custom Spaces need the `/clone_batch` extension for `--text-file`; its source
and rebuild instructions are in
[`deploy/huggingface/omnivoice`](https://github.com/hkchengrex/home-assistant-voice-io/tree/main/deploy/huggingface/omnivoice).

## Authentication

The default Space is public. Sign in with `hf auth login` to use your account's
ZeroGPU allowance, including PRO benefits. The client uses the saved login when
`HF_TOKEN` is not set. An environment token overrides the saved login, so an old
`HF_TOKEN` can continue to select an older credential.

For an explicit environment token:

=== "Windows PowerShell"

    ```powershell
    $env:HF_TOKEN = "your-token"
    ```

=== "macOS / Linux"

    ```bash
    export HF_TOKEN="your-token"
    ```

Do not put actual tokens in `commands.toml`, scripts, shell history, or the
repository. Prefer the interactive login. Running generation does not require
repository write permission; deploying changes to the Space does.

## Python API

```python
from pathlib import Path

from ha_voice import CloneSettings, HuggingFaceSpaceVoiceCloner

cloner = HuggingFaceSpaceVoiceCloner()
result = cloner.clone_to_library(
    text="Welcome home",
    reference_audio=Path("voice-data/reference.wav"),
    assets_dir=Path("voice-data/assets"),
    response_prefix="welcome",
    settings=CloneSettings(language="English"),
    consent_to_upload=True,
)

print(result.source_path)
```

For several clips in one response group:

```python
batch = cloner.clone_batch_to_library(
    texts=["Welcome home.", "Good to see you again."],
    reference_audio=Path("voice-data/reference.wav"),
    assets_dir=Path("voice-data/assets"),
    response_prefix="welcome",
    settings=CloneSettings(language="English"),
    batch_size=4,
    consent_to_upload=True,
)
print(batch.source_paths)
print(batch.remote_metrics)
```

The Space's `/clone_batch` API returns a list of downloadable PCM16 WAV files
and a JSON report containing `prompt_seconds`, `generation_seconds`, and
`gpu_task_seconds`. These server timings exclude queueing and network transfers.
The report also includes the GPU name, sample rate, and per-file duration/index.

For a private Space, pass a token read from your environment to `HuggingFaceSpaceVoiceCloner(token=...)`.

## Quality controls

The common controls are available as command options:

- `--steps`: 4–64 inference steps; 32 is the quality/speed default.
- `--guidance-scale`: 0–4; 2.0 is the default.
- `--speed`: 0.5–1.5 unless a fixed duration is supplied.
- `--duration`: optional fixed output duration in seconds.
- `--instruct`: optional voice direction supported by OmniVoice.
- `--attempts`: total attempts after transient ZeroGPU or connection failures; the default is 2.

The first request may take longer if the Space needs to wake or load the model.

## Batch benchmark

Measured on 2026-08-31 UTC using four short English lines, 32 inference steps,
a 4.87-second synthetic reference with a supplied transcript, and three runs per
mode after one excluded single-request warm-up. Test order rotated between runs.
The GPU reported **NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb**.
This is Hugging Face ZeroGPU hardware, not the Surface Go used for recognition.

| Mode, four clips | Median total request time | Median model generation time |
| --- | ---: | ---: |
| Four separate requests | 14.29 s | Not instrumented |
| One request, shared reference, generated one at a time | 10.93 s | 3.27 s |
| One request, all four generated together | 8.12 s | 1.25 s |

Batching was **1.76× faster end-to-end than four separate requests**. Model
generation was **2.61× faster than shared-reference sequential generation**.
Total time includes uploads,
queueing, GPU startup, and downloads, but not client initialization or local
asset publishing. Each clip was about 2.1–2.6 seconds long. Individual batch
requests ranged from 4.72 to 10.64 seconds; shared GPU availability matters.

This small test validates speed and usable PCM files, not voice similarity or
quality across languages and long texts. Start with batches of four short lines;
larger or uneven-length batches may waste padding or need more memory. GPU
assignments and queue times can change, so these are observations, not guarantees.

[Raw timings](https://github.com/hkchengrex/home-assistant-voice-io/blob/main/benchmarks/omnivoice-batch-2026-08-31.json)
and the repeatable
[benchmark script](https://github.com/hkchengrex/home-assistant-voice-io/blob/main/scripts/benchmark_voice_generation.py)
are in the repository. Run the script with `--reference`, `--reference-text`,
`--output`, and `--confirm-upload`; benchmark audio stays in your output folder.
