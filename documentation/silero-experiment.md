# Silero continuous audio experiment

The optional `silero` extra installs ONNX Runtime 1.22.1. Inference uses one CPU
thread and requires no PyTorch, browser or cloud connection. Live listener defaults
are unchanged; Silero is an offline experiment only.

Download the official model on your trusted development machine from:

https://raw.githubusercontent.com/snakers4/silero-vad/867c2aa692646a1f1de3e94a15c9dd9f614c0acb/src/silero_vad/data/silero_vad.onnx

The adapter verifies SHA256
`1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
The model is provided by [Silero under its MIT license](https://github.com/snakers4/silero-vad/blob/867c2aa692646a1f1de3e94a15c9dd9f614c0acb/LICENSE).
Do not substitute another model without validating its interface and updating the
pin. Runtime compatibility must be tested on the actual target CPU.

```sh
uv sync --frozen --extra silero
uv run --frozen --extra silero python -m ha_voice.continuous_vad_experiment \
  --audio /private/session/continuous.wav \
  --model /private/models/silero_vad.onnx \
  --config /private/commands.toml --recordings /private/recordings \
  --mute-intervals /private/session/mute-intervals.json \
  --output /private/session/replay.json
```

Pass the deployment's `--min-rms` and `--noise-multiplier` to reproduce its energy
baseline. A mute file is an array of objects with `start`, `end` (seconds from
capture start) and `kind` (`wake` or `command`). It must come from observed speaker
playback intervals. Omit it for a raw-audio sensitivity check; speaker responses
can then be misinterpreted as speech. Never publish private audio or report files
containing private template paths, hashes or labels in this repository.

## What is compared

Both detectors receive the same continuous 16 kHz mono recording in 20 ms frames.
The Silero bridge waits for complete 512-sample windows and retains 64 samples of
context and recurrent state. It is causal: no future frames are used. A probability
is held between model updates, adding up to 32 ms update delay. Thresholds are 0.5
for speech onset and 0.35 for release. There is no extra RMS gate for Silero.

Both use the existing segmenter's 80 ms onset, 300 ms endpoint, 700 ms look-back,
2.5 second cap, final trimming, and the frozen wake/command templates and
thresholds. Duration-limited wake captures are rejected as in the listener.
Every segment is also checked against a separate wake-only gate for diagnosis.
The pipeline gate uses recording time so accelerated processing does not extend
its command timeout. Observed playback intervals reset segmentation, and an
already accepted wake's command window is refreshed after its observed reply.

## Interpretation limits

This reuses actual listener algorithms but is not a complete live simulation.
Matching queue delay is not modeled, and playback masking follows the original
listener's timing rather than a candidate's hypothetical timing. Recordings
contain the original listener's speaker audio. A candidate that detects a wake
later can be affected by the original listener's already-started response.
Check that replayed baseline events agree with the actual listener before using
paired acceptance counts to claim improvement. Report raw and masked results
when this affects conclusions.

Frame processing time is not user response latency. Human-marked speech endpoints
are needed for that. Small prompted sessions establish specific success/failure
cases, not general recall, precision or false wakes per hour. Keep session-level
holdouts for later evaluation. Compare actual wake and command outcomes and
inspect split/truncated speech; do not use segment counts alone as success.
