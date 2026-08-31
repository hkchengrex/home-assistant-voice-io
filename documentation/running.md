# Run continuously

Continuous mode watches the microphone, detects an utterance, classifies it, and optionally plays a local response. Your application can attach an action handler for accepted commands.

## Wake-phrase mode

This is the normal mode for an always-on device. Train the `_start_phrase` set, enable `[start_phrase]` in `commands.toml`, then run:

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings run --assets voice-data/assets
```

The listener waits for the start phrase, plays its response if configured, and opens a short window for one command.

Before relying on it:

1. Record at least the configured `target_samples` for the start phrase.
2. Record unrelated speech for `_not_command` and several non-wake phrases for rejection.
3. Test from realistic positions and with typical background noise.
4. Review recent triggers in the Studio and promote false triggers as negative examples.

## Direct mode

Direct mode classifies every detected utterance and disables actions. It is useful for testing commands or collecting fresh examples:

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings run --direct
```

To save five new test utterances as examples for one command:

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings run --direct --capture-command lights_on --capture-count 5
```

## Input and output devices

Pass a device number or name with `--input-device` and `--output-device`. Run `voice-io devices` to see available inputs.

If the listener misses quiet speech, inspect the microphone first. `--min-rms` and `--noise-multiplier` tune voice activity detection, but very aggressive settings can increase false triggers.

## Local responses

The `[responses]` section maps a response group to an audio filename prefix. Put matching audio files in the assets folder passed to `run`. A command can name a response group, or leave it blank for no local playback.

Responses are optional and separate from the external action. Your handler can also suppress feedback when an action fails or already provides its own confirmation.
