# Deploy on your platform

First get recognition working interactively. Then run the same command as a background user service with an explicit config, recordings folder, and assets folder.

The repository includes starting points under `deploy/`:

| Platform | Example | Best for |
| --- | --- | --- |
| Linux | `deploy/systemd/voice-io.service` | A user-level systemd service |
| macOS | `deploy/launchd/io.havoice.voiceio.plist` | A per-user launch agent |
| Windows | `deploy/windows/run-voice-io.ps1` | A Task Scheduler action |

## Common deployment shape

Keep code and personal data separate:

```text
application checkout or installed package
private data folder/
├── commands.toml
├── recordings/
└── assets/
```

Use absolute paths in the service definition. Run it as an ordinary user with access to the audio device; root or administrator access should not be necessary.

## Linux

Copy the example unit to `~/.config/systemd/user/`, replace its working paths, then enable it as a user service. PipeWire or PulseAudio must be available in the same user session.

## macOS

Copy the example property list to `~/Library/LaunchAgents/`, replace every placeholder, and make sure the executable has microphone permission. Test the exact command in Terminal before loading the agent.

## Windows

Adjust the example PowerShell script to your checkout and private data folder. In Task Scheduler, run it only when the intended user is logged on so the process can use that user's audio device.

## Safe updates

The package and CLI are named `voice-io`. When upgrading an existing installation,
stop its listener and Studio first, install into a fresh virtual environment, and
update executable paths and service names to the examples above. Disable the
previous service units before enabling replacements so only one listener captures
audio. Keep configuration, recordings, and response assets in their existing data
folders; a package rename does not require renaming or regenerating them.

Pin deployments to a reviewed release or commit. Before switching versions:

1. Back up private configuration and recordings.
2. Install the new package in a fresh virtual environment.
3. Run a one-off recognition test.
4. Restart the service and inspect its first few accepted and rejected events.

Never copy repository credentials or private SSH keys to an always-on voice device. Deploy from a trusted development machine.
