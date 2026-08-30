# Training Studio

The Studio is a local browser interface for building and improving a voice library. It runs on your machine; it is not a hosted recording service.

Start it with:

```shell
local-voice --config voice-data/commands.toml --recordings voice-data/recordings studio
```

Then open `http://127.0.0.1:8765`.

## Build a useful recording set

Choose a command from the left side, select your microphone, and record examples. You can keep or retry each take, replay saved takes, replace weak recordings, and remove mistakes.

For hands-free collection, choose **Record a full set**. A chime starts each take and the Studio saves it automatically.

Good training examples include normal variation:

- Stand where you will actually use the command.
- Vary your distance and speaking pace a little.
- Include normal room noise, but avoid clipping or very low volume.
- Start with about 10 recordings per command and 15 for the wake phrase.

## Add commands

Choose **Add command**, then provide a stable command name, the phrase to display, and a short description. The Studio adds it to your private `commands.toml` and creates its recording set.

Use `intent_group` when two different spoken phrases should trigger the same logical action. That keeps aliases from competing against each other during acceptance.

## Test without triggering actions

**One-off test** records and classifies one phrase. **Start continuous** listens repeatedly. Both Studio test modes deliberately avoid sending external actions, so you can evaluate recognition safely.

## Compare microphones

**Compare microphones** alternates between two input devices while you hold the same position. Replay each pair and choose the device with clearer, more consistent speech—not simply the loudest signal.

## Review false triggers

In wake-phrase mode, the listener keeps a bounded set of recent accepted triggers. The Studio lets you replay them, dismiss expected results, or promote false triggers into negative examples. Those examples help the recognizer reject similar sounds in the future.

!!! warning "The Studio can change your library"
    Adding or removing commands and takes edits files in your data folder. Back up `commands.toml` and recordings privately if the library is important.
