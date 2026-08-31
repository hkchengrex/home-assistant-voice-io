# Home Assistant Voice IO

Home Assistant Voice IO (HA Voice IO) is an offline, language-independent voice-command system.
It learns commands from WAV recordings, compares speech with PCEN/MFCC features
and dynamic time warping, provides a browser-based training studio, and can run
an always-on start-phrase listener. It does not transcribe speech or require a
cloud service.

**[Read the documentation](https://hkchengrex.com/home-assistant-voice-io/)**
for the guided setup, training, integration, and deployment guides.

## What it does

- Teach it custom voice commands by recording examples.
- Recognize those commands locally without sending audio to the cloud.
- Use a wake phrase before commands.
- Manage and test commands in a browser-based training studio.
- Connect recognized commands to Home Assistant or another system.
- Play spoken or audio responses.
- Generate cloned response clips through the optional OmniVoice Space client.
- Run continuously on Windows, macOS, or Linux.
- Review failed recognitions and improve the training set.

It recognizes phrases you train; it is not general speech-to-text or an
open-ended voice assistant.

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

## Benchmark

The reference low-power deployment is a first-generation Microsoft Surface Go
with an Intel Pentium Gold 4415Y at 1.60 GHz (2 cores / 4 threads), running the
native matcher on Python 3.12.

- **Recognition response:** 86.9 ms median and 91.9 ms p95 over 100 measured
  runs.
- **Workload:** one 0.72-second recorded command matched against 123 templates,
  after 10 warm-up runs.
- **Measured work:** silence trimming, feature extraction, DTW matching, and
  command classification after captured audio is available. Microphone capture,
  response playback, and external automation latency are excluded.
- **Live observation:** the two most recent successful commands in the running
  deployment reached the Home Assistant action in 150–190 ms after the detected
  command ended (170 ms median). This is a small operational sample, not the
  controlled benchmark above.

Results will vary with phrase length, template count, selected backend, system
load, and external automation latency.

## Install

The Python distribution and command are named `voice-io`. Python imports remain
`ha_voice` for compatibility with existing integrations. Until a package release
is published, install from this repository checkout:

```shell
python -m pip install -e ".[capture]"
```

Add the optional Hugging Face client when generating cloned response audio:

```shell
python -m pip install -e ".[capture,voice-clone]"
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
voice-io devices
voice-io --config voice-data/commands.toml --recordings voice-data/recordings studio
```

Train at least two commands, calibrate, and run:

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings calibrate
voice-io --config voice-data/commands.toml --recordings voice-data/recordings run --assets voice-data/assets
```

The CLI performs recognition and local responses but intentionally has no
external action implementation. Applications call `ha_voice.cli.main` with a
`CommandHandler`, or use the lower-level audio, matcher, listener, and Studio
APIs directly.

## Generate response audio

The optional OmniVoice integration can clone a response from a short reference
recording and publish it into an existing response group:

```shell
voice-io --config voice-data/commands.toml clone-response \
  --response welcome \
  --text "Welcome home" \
  --reference voice-data/reference.wav \
  --assets voice-data/assets \
  --confirm-upload
```

The reference recording is uploaded to the configured Hugging Face Space. The
default `hkchengrex/OmniVoice` Space is currently public and does not require a
token. If authentication is enabled later, set `HF_TOKEN` in the environment;
the token is never stored in the voice configuration.

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
