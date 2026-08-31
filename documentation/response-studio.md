# Response generation Studio

Use **Response generation** in the local Studio to write replies, generate alternatives,
and choose which takes enter playback. This is separate from command-recognition training.

Install the optional client from the repository checkout:

```shell
python -m pip install -e ".[capture,voice-clone]"
voice-io --config voice-data/commands.toml --recordings voice-data/recordings studio
```

Open `http://127.0.0.1:8765/responses`. The interface works on Windows, macOS, and Linux.
The generation page does not require a microphone. Keep Studio on its default loopback
address; it is a local tool, not an authenticated public web service.

## Example replies

A new workspace starts with these editable examples. Commands sharing a response group
share its reply list. Wake acknowledgements and event responses also have examples.

| Example command or event | Response group | Reply phrases |
| --- | --- | --- |
| Wake phrase | `start` | Yes? · I'm listening. · Go ahead. · How can I help? |
| Welcome event | `welcome` | Welcome home. · Good to see you. · Welcome back. · Make yourself comfortable. |
| Morning event | `morning` | Good morning. · Morning! · Have a good day. · Ready for a new day? |
| `lights_on`, `day_routine` | `lights_on` | Done. · Lights on. · The lights are on. · There you go. |
| `lights_off` | `lights_off` | Done. · Lights off. · The lights are off. · All switched off. |
| `good_night` | `good_night` | Good night. · Sleep well. · Sweet dreams. · Rest well. See you tomorrow. |
| `byebye`, `byebye_native` | `bye` | Bye! · See you later. · Take care. · Have a good one. |
| `nothing` | `nothing` | Okay. · No problem. · All right. · Let me know if you need anything. |

Choose a response group, then add or edit phrases. **Archive** hides a phrase from generation
without deleting its takes; **Show archived** lets you restore it. Custom groups without bundled
examples start empty. Editing text never relabels old audio: each take retains its original text.

## Generate and compare

1. Save a reference recording: 1–20 seconds, at most 10 MB. WAV (including 24-bit,
   32-bit, and floating-point), MP3, M4A/AAC, FLAC, OGG, WebM, and AIFF are supported.
   Studio converts it locally to mono, 24 kHz, 16-bit PCM WAV; no manual conversion
   is needed. Conversion preserves timing and does not trim the recording.
   Enter its exact transcript, or leave the transcript empty for automatic transcription.
2. Set the voice controls and check the permission box. Saving a reference stays local;
   generating sends that recording and the selected text to the OmniVoice Hugging Face Space.
3. Select one or more phrases and choose **Takes per phrase** (1–8). For example, three
   phrases with four takes each produce twelve candidates.
4. Choose **Generate selected phrases**. Each job supports up to 64 takes. Studio splits it
   into requests of at most eight clips and 1200 characters, using the selected GPU batch size.
5. Play each take, then **Keep** or **Reject** it. Select several takes for bulk review.
   Pause any live listener before playing previews near its microphone.
6. **Regenerate** a rejected take, or select several and choose **Regenerate rejected**.
   Each gets one new draft using its original text and your current settings/reference.
   The rejected audio stays available for comparison. Reject a replacement to try again.
7. **Publish kept takes** adds all kept, unpublished takes in the current group to playback.
   Use **Remove from playback** to undo publication; the draft and a local backup remain.

Pending and rejected takes never enter playback. Existing response clips are not replaced.
The player can choose among existing clips and your newly published takes. A managed listener
is refreshed after publication; otherwise restart the listener yourself.

Reference conversion uses PyAV, included in the `voice-clone` extra. Its wheels
bundle the audio codecs on supported Windows, macOS, and Linux systems; a separate
FFmpeg executable is not required. The converter accepts one audio track with up
to eight channels, at 8–384 kHz. Invalid or over-limit uploads leave the previous
reference unchanged. The saved preview and the reference sent for generation are
the same converted WAV. This does not change the recognition recording format.

## Voice controls

| Control | Range or behavior |
| --- | --- |
| Speed | 0.5–1.5; 1 is normal |
| CFG scale | 0–4; guidance strength, default 2 |
| Inference steps | 4–64; fewer steps generally reduce generation time |
| Language | Auto, or an OmniVoice-supported language |
| Voice direction | Optional voice attributes passed to OmniVoice |
| Fixed duration | Empty for automatic timing, or up to 15 seconds per clip; overrides speed |
| Denoise | Request denoising from the Space |
| Preprocess reference | Request reference preprocessing |
| Postprocess output | Request model output postprocessing |
| GPU batch size | 1–8, default 4; lower it if GPU memory is insufficient |

OmniVoice currently exposes one CFG guidance scale for cloning. Each take shows its saved
parameters so you can compare runs. Studio also normalizes generated WAVs for the local
response player. **Save voice settings** saves controls without generating anything.

Generation is not guaranteed to improve with higher CFG or more steps; compare a few short
replies. [Authentication and batch performance](response-generation.md#authentication)
are the same as for the command-line client. Credentials stay on the server and are not
entered into or returned by this UI.

## Jobs, storage, and recovery

Closing the browser does not stop a running job while the Studio server remains running.
**Stop after this batch** retains the in-flight batch and prevents subsequent requests.
If a request fails, completed earlier batches remain saved. **Retry remaining takes** submits
only the unfinished entries, with that job's original reference and settings. It does not
regenerate already completed takes. After a server restart, interrupted jobs require an
explicit retry; there is no automatic cloud generation.

Defaults beside the recordings directory:

```text
voice-data/
  commands.toml
  recordings/
  assets/                         # published playback files
  response-studio/
    workspace.json                # phrases, settings, reviews, jobs
    references/                   # current and past reference recordings
    candidates/                   # all saved takes, including rejected ones
    unpublished/                  # backups of removed playback copies
```

Use `studio --assets PATH --response-workspace PATH` to override these locations. Run only
one Studio server per response workspace, and avoid concurrent publishers to the same assets
directory. Back up these folders privately; references, transcripts, generated audio, and
review history do not belong in the public source repository. The default `response-studio/`
directory is ignored by Git.

Published files use `<prefix>_studio_<id>.wav`. Studio tracks the files it creates and refuses
to remove a playback copy changed outside Studio. Command-line library publishing preserves
these Studio-managed clips.
