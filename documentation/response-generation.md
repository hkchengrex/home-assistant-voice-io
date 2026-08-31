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

## Authentication

The default `hkchengrex/OmniVoice` Space is public and currently needs no token. If it becomes private or gated, set a Hugging Face access token in the environment:

=== "Windows PowerShell"

    ```powershell
    $env:HF_TOKEN = "your-token"
    ```

=== "macOS / Linux"

    ```bash
    export HF_TOKEN="your-token"
    ```

Do not put the token in `commands.toml`, scripts, shell history, or the repository. The client reads `HF_TOKEN` at runtime and does not save it.

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
