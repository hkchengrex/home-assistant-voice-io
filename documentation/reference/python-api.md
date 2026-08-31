# Python API

The stable top-level API exposes configuration, templates, classification results, and the automation handler contract:

```python
from ha_voice import (
    ActionResult,
    AppConfig,
    BatchCloneResult,
    CloneResult,
    CloneSettings,
    GeneratedBatch,
    CommandConfig,
    CommandHandler,
    MatchResult,
    RecognizerConfig,
    Template,
    HuggingFaceSpaceVoiceCloner,
    VoiceGenerationError,
    classify,
    load_config,
    load_templates,
)
```

## Configuration and templates

```python
from pathlib import Path
from ha_voice import load_config, load_templates

config = load_config(Path("voice-data/commands.toml"))
templates = load_templates(
    Path("voice-data/recordings"),
    config.recognizer.sample_rate,
)
```

`load_config` validates names, ranges, references, and the presence of at least one command. `load_templates` reads the private WAV library and extracts the data needed for matching.

## Classification

`classify` accepts command features, a template collection, and the recognizer thresholds. It returns a `MatchResult` with the accepted command, score, margin, per-command scores, and acceptance flag.

For most applications, the continuous CLI already owns capture, wake gating, feedback, and diagnostics. Supply a `CommandHandler` to `ha_voice.cli.main` rather than reimplementing that loop.

## Compatibility boundary

Install the `voice-io` distribution and continue to import `ha_voice`. The package
rename does not change the Python import namespace or application contracts.

The `ha_voice` import namespace and top-level action/configuration contracts are the intended public boundary. Lower-level audio, feature, Studio, listener, and service modules are available for advanced use but may evolve more quickly before version 1.0.

The Hugging Face voice-cloning API is optional. Importing its public classes does not require `gradio-client`; creating a live client does. Install the `voice-clone` extra before connecting to a Space.

## Generate drafts without publishing

`HuggingFaceSpaceVoiceCloner.generate_batch(...)` accepts the same voice settings,
reference, text list, and GPU batch size as `clone_batch_to_library(...)`, but it does
not change the playback library. It returns `GeneratedBatch(audio_paths, remote_metrics)`.
The audio paths point to client downloads; copy them into your own storage to retain them.
Explicit `consent_to_upload=True` is required. Studio uses this path to review takes
before publication; command-line `clone-response` continues to publish directly.
