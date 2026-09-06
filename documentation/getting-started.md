# Train your first command

This guide covers installation, recording training examples, and testing a command.

## What you need

- Windows, macOS, or Linux
- Python 3.11–3.13
- A microphone
- A quiet place for the first recording set

The package is not on PyPI yet. Install it from a local checkout.

## 1. Create an environment

=== "Windows PowerShell"

    ```powershell
    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -e ".[capture]"
    ```

=== "macOS / Linux"

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[capture]"
    ```

The `capture` extra installs microphone support. Matching existing WAV files only needs the base package.

## 2. Make a private data folder

Copy the example `commands.toml` into a folder that will hold your personal configuration:

```text
voice-data/
├── commands.toml
├── recordings/
└── assets/
```

Keep this folder out of a public repository. Voice recordings are personal data.

## 3. Check your microphone

```shell
voice-io devices
```

Note the number or name of the microphone you want to use. If your default device is correct, you can leave the device option unset.

## 4. Open the Training Studio

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings studio
```

Open `http://127.0.0.1:8765`. Choose a phrase, select a microphone, and record at least two examples. Ten varied examples per phrase is a better starting point for everyday use.

## 5. Test one command

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings listen
```

A successful test prints the best match, its intent, and the confidence margin. A rejection is safe: no command is sent.

## 6. Calibrate

After training at least two commands with two examples each:

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings calibrate
```

Use the suggested midpoint as a starting value for `recognizer.max_distance`. If command classes overlap, add more varied recordings or choose phrases that sound less alike.

## Next

- [Learn the Training Studio](training-studio.md)
- [Run with a wake phrase](running.md)
- [Connect accepted commands to an automation](integrations.md)
