# Troubleshooting

## No microphone appears

Run `local-voice devices`. If the list is empty:

- Confirm the operating system has granted microphone access to the terminal or Python process.
- On Linux, confirm PipeWire or PulseAudio is running for the same user.
- Reinstall the capture extra: `python -m pip install -e ".[capture]"`.
- Disconnect and reconnect USB audio devices, then list devices again.

## The right phrase is rejected

Start with the recordings before changing thresholds:

1. Replay several saved takes in the Studio.
2. Remove clipped, cut-off, unusually quiet, or mislabeled takes.
3. Add examples from the real speaking distance and room.
4. Run `local-voice calibrate` again.

Then consider raising `max_distance` slightly or lowering `min_margin`. Looser settings can also increase false accepts, so test unrelated speech afterward.

## The wrong command wins

Choose phrases that sound more distinct, add varied examples to both commands, and remove ambiguous takes. If the phrases are alternate wording for the same action, give them the same `intent_group`.

## The wake phrase triggers accidentally

Open **Recent triggers** in the Studio, replay the capture, and promote genuine false triggers as negative examples. Also record varied `_not_command` examples and avoid a wake phrase that resembles common background speech.

## Continuous listening stops

Run the exact service command interactively and watch the output. Common causes are a missing audio device after reboot, a config path relative to the wrong working directory, or a response asset that cannot be played.

## Matching is slower than expected

Check whether the optional native matcher built successfully. Performance also scales with the number and length of templates. Keep useful variation, but remove duplicates and poor-quality takes rather than growing the library indefinitely.

## Get useful diagnostic information

When reporting a problem, include:

- Operating system and Python version
- CPU model
- Template counts from `local-voice ... templates`
- Whether matching uses the native or NumPy backend
- Sanitized configuration thresholds
- The printed distance, margin, duration, and matching time

Do not attach personal recordings publicly unless every speaker has explicitly agreed.
