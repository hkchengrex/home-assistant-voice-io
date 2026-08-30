# Local Voice Pipeline

Local Voice Pipeline recognizes short voice commands from recorded examples.
Recognition runs locally on Windows, macOS, and Linux. It does not transcribe
arbitrary speech or answer open-ended questions.

## Overview

- Record training examples and review them in the [Training Studio](training-studio.md).
- Recognize trained commands, optionally after a [wake phrase](running.md).
- Send recognized intent names to [Home Assistant or another application](integrations.md).
- Play local response clips or [generate clips from a reference voice](response-generation.md)
  using the optional OmniVoice integration.
- Run the listener as a service using the [deployment examples](deployment.md).

Recognition does not require a cloud service. Optional voice cloning uploads
the reference recording and response text to a Hugging Face Space; see
[privacy and data ownership](privacy.md).

## Setup and usage

Start with the [getting started guide](getting-started.md) to install the package,
record examples, and test a command.

- [Configuration](configuration.md): commands, recognition settings, and responses.
- [Command line](reference/cli.md): available commands and options.
- [Python API](reference/python-api.md): integrating the library into an application.
- [Troubleshooting](troubleshooting.md): microphone, recognition, and service issues.

## Benchmark

On a first-generation Surface Go with an Intel Pentium Gold 4415Y, matching a
0.72-second command against 123 templates took 86.9 ms median and 91.9 ms p95.
These measurements exclude microphone capture, response playback, and automation
latency. See the [benchmark method and results](benchmark.md).

## Source

The [source repository](https://github.com/hkchengrex/home-assistant-voice-io)
contains the library, tests, and deployment examples. The library is MIT licensed.
