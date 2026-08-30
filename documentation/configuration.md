# Configuration

`commands.toml` describes what can be recognized and how the local listener behaves. It should contain no credentials or private automation endpoints.

## Minimal example

```toml
[recognizer]
sample_rate = 16000
max_distance = 3.6
min_margin = 0.03
top_k = 2

[start_phrase]
enabled = true
utterance = "Hello house"
target_samples = 15
command_timeout_seconds = 3.0

[commands.lights_on]
description = "Turn on the room lights"
utterance = "Lights on"
target_samples = 10
intent_group = "day_lighting"
response = "lights_on"

[responses]
lights_on = "lights_on"
```

## Recognition settings

| Setting | Purpose | Default |
| --- | --- | --- |
| `sample_rate` | Audio sample rate used for recordings and matching. | `16000` |
| `max_distance` | Largest template distance that can be accepted. Lower is stricter. | `4.0` |
| `min_margin` | Required separation from the runner-up. Higher is stricter. | `0.12` |
| `top_k` | Number of nearest templates combined for each command. | `3` |

Use `local-voice calibrate` to choose a sensible starting distance for your own recordings.

## Commands

Each `[commands.name]` entry has:

- `utterance`: the phrase shown while recording.
- `description`: a human-readable explanation.
- `target_samples`: the desired number of examples.
- `intent_group`: the logical action shared by alternate phrases.
- `response`: an optional key from `[responses]`.

Names, intent groups, and response keys may contain letters, numbers, `_`, and `-`.

## Start phrase

The `[start_phrase]` section has its own distance, margin, and `top_k` because wake recognition is a different decision from command recognition. It also controls:

- `command_timeout_seconds`: how long to wait for the command after waking.
- `min_audio_seconds` and `max_audio_seconds`: accepted wake-phrase duration.
- `min_command_audio_seconds`: shortest command evaluated after waking.
- `response`: optional local acknowledgment.

## Calibration and negative examples

`[calibration]` describes unrelated phrases used to measure rejection. Record varied speech under `_not_command`; these are not commands and never trigger an action.

## Local control events

`[control_events]` can map loopback-only HTTP paths to response groups when `run --control-port` is enabled:

```toml
[control_events]
"/welcome" = "welcome"
```

The server binds to `127.0.0.1`. Treat it as local process coordination, not a network API.
