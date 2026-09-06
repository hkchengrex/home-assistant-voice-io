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

Use `voice-io calibrate` to choose a sensible starting distance for your own recordings.

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
## Command PCEN tuning

PCEN reduces the impact of steady room noise, microphone gain, and speaker
distance before matching a command. Leave its settings at their defaults unless
recordings from the actual listening position show a repeatable problem. These
optional `[recognizer]` keys apply consistently to command templates, CLI
queries, Studio, and the command phase after a wake; wake-phrase features are
unchanged.

```toml
[recognizer]
# Defaults; change only one value at a time.
pcen_smoothing = 0.05
pcen_alpha = 0.98
```

| Setting | Default and range | Effect of increasing it |
| --- | --- | --- |
| `pcen_smoothing` | `0.05`; greater than 0 and at most 1 | Makes the energy baseline follow new sound levels more quickly. Lower values adapt more gradually. |
| `pcen_alpha` | `0.98`; between 0 and 1 | Applies stronger normalization for recent per-frequency energy levels. |

Tune safely:

1. Keep the defaults as a baseline and collect examples in the location and with
   the microphone used for normal listening.
2. Change one setting by a small amount, restart Studio or the listener so it
   reloads the configuration and regenerates templates from the WAV files.
3. Run `voice-io calibrate` and repeat real listening tests. Recheck
   `recognizer.max_distance` and `recognizer.min_margin`, because PCEN changes
   the distances on which those acceptance settings depend.
4. Keep a change only when it improves the accepted/rejected behavior on
   recordings not used to choose it. Otherwise restore the defaults.

For a reproducible offline comparison of preset PCEN candidates on a labeled
recording bank, see the [PCEN parameter sweep](pcen-sweep.md). Its results are
experimental rather than universal recommendations.
