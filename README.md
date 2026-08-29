# Local Voice Pipeline

Local Voice Pipeline is an offline, language-independent voice-command system.
It learns commands from WAV recordings, compares speech with PCEN/MFCC features
and dynamic time warping, provides a browser-based training studio, and can run
an always-on start-phrase listener. It does not transcribe speech or require a
cloud service.

The library is automation-neutral. Applications attach a `CommandHandler` to
accepted intent names; response groups and local control events are declared in
TOML. A separate deployment can therefore connect the same recognizer to Home
Assistant, another automation system, or a custom program.

## Platforms

The package is tested on Windows, macOS, and Linux with Python 3.11–3.13. Audio
capture uses PortAudio through the optional `sounddevice` dependency. Matching
and WAV-file workflows require only NumPy. An optional C acceleration module is
built when a supported compiler is available; the NumPy implementation remains
the portable fallback.

## Install

```shell
python -m pip install "local-voice-pipeline[capture]"
```

For development:

```shell
python -m pip install -e ".[dev]"
pytest
```

Copy `commands.toml` to an application data directory, then keep recordings and
responses beside that private configuration:

```text
voice-data/
  commands.toml
  recordings/
  assets/
```

List devices and open the training studio:

```shell
local-voice devices
local-voice --config voice-data/commands.toml --recordings voice-data/recordings studio
```

Train at least two commands, calibrate, and run:

```shell
local-voice --config voice-data/commands.toml --recordings voice-data/recordings calibrate
local-voice --config voice-data/commands.toml --recordings voice-data/recordings run --assets voice-data/assets
```

The CLI performs recognition and local responses but intentionally has no
external action implementation. Applications call `ha_voice.cli.main` with a
`CommandHandler`, or use the lower-level audio, matcher, listener, and Studio
APIs directly.

## Configuration boundary

- `intent_group` combines alternate phrases that cause the same logical effect,
  preventing aliases from reducing one another's acceptance margin.
- `response` points at a named entry in `[responses]`.
- `[control_events]` maps loopback POST paths to response groups.
- No tokens, webhook IDs, entity IDs, hostnames, or user recordings belong in
  this repository.

## Deployment

Portable examples are under `deploy/` for Linux systemd user services, macOS
launchd, and Windows Task Scheduler. They use placeholders and external data
directories; copy and customize them in the consuming deployment repository.

## Privacy

Voice recordings are biometric-adjacent personal data. `recordings/` and
`assets/` are ignored by default. Use synthetic or explicitly redistributable
fixtures in public tests, and keep real recordings in a private backup.

## License

MIT
