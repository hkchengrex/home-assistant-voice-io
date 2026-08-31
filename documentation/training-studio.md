# Training Studio

The Studio is a local browser interface for building and improving a voice library. It runs on your machine; it is not a hosted recording service.

Start it with:

```shell
voice-io --config voice-data/commands.toml --recordings voice-data/recordings studio
```

Then open `http://127.0.0.1:8765`.

For spoken replies rather than recognition examples, choose **Response generation**
in the header. Its [separate workspace](response-studio.md) supports editable reply lists,
multiple generated takes, keep/reject review, regeneration, and OmniVoice controls.

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

## Review missed commands and false triggers

In **Recent triggers**, replay a clip, choose its correct label, and click **Teach
as…**. Choose a command or the wake phrase for a missed/misrecognized example;
choose **Not a command** or **Not the wake phrase** for a false trigger. Each clip
is labeled separately, so correcting a command does not reject a valid wake phrase.
Teaching saves one training take and reloads a managed listener; it never executes
the labeled command. Stop Studio continuous testing before teaching a clip.

An accepted wake phrase and its following command are normally saved, including
rejected commands. To collect missed wake phrases, enable **Capture missed attempts
(5 min)**. This opt-in also retains speech rejected before the command window.
It uses the already-running listener; it does not start a microphone itself or change
recognition thresholds. Without a live listener, use Studio's **Start continuous**.
Normal live voice actions remain active; Studio continuous tests do not run actions.

Capture stops automatically after five minutes or when you click **Stop capturing
missed attempts**. Closing the page does not extend the deadline. Only the latest
20 events are kept in the rolling queue. Audio below the speech-detection threshold
cannot appear here: use **Record sample** if an attempt never appears.

Labeling leaves the original clips in the queue and saves the reviewed take under
`recordings/<label>/`, with provenance in `recordings/_review_metadata/`.
Repeated submissions do not duplicate examples. **Dismiss** only removes the queue
entry, not a taught example; remove incorrect training takes through the normal
saved-take controls. Keep reviewed audio and metadata in your private backups.

!!! warning "The Studio can change your library"
    Adding or removing commands and takes edits files in your data folder. Back up `commands.toml` and recordings privately if the library is important.
