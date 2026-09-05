# Speech detection experiment

This optional offline experiment compares the existing energy detector with
WebRTC modes 0 through 3. It never opens a microphone, runs recognition, plays
audio, or invokes an action. Live listener and Studio defaults remain unchanged.

## Optional live trial

After installing `voice-io[capture,vad]`, select the detector explicitly with
`voice-io run --vad webrtc --vad-mode 1`. The default remains `--vad energy`.
The detector runs locally on the listener host, without a browser or a network
service. The selection survives segmenter resets after response playback.
RMS settings apply to the energy backend; WebRTC is not additionally gated by
those thresholds. Onset, endpoint timing and final trimming remain unchanged.
Revert by removing these two options or selecting `--vad energy` and restarting
the listener. Test real noise: WebRTC can classify tones and hum as speech.

Install with `uv sync --extra dev --extra vad`. Run:

```sh
uv run --extra vad python -m ha_voice.vad_experiment \
  --recordings /path/to/private/recordings \
  --include-label wake --include-label lights_on \
  --min-rms 0.004 --noise-multiplier 3.0 \
  --output /path/to/private/results/vad.json
```

Pass the energy thresholds used by the deployment. Explicitly list each saved
speech directory; do not include arbitrary backup or diagnostic directories.
Run on the intended hardware with its normal services running. Keep the output
private: it includes recording filenames and hashes. Do not add personal data
to this public repository.

## Interpretation

All backends receive identical 16 kHz audio in 20 ms blocks, a 1.2 second
calibration prelude and one second of trailing background. The test retains
80 ms onset, 300 ms endpoint silence, 700 ms look-back, the 2.5 second duration
cap, and the existing energy-based final trimming. WebRTC is not gated by RMS.
Modes differ only in WebRTC aggressiveness; mode 3 rejects non-speech most
aggressively. Detector state is new for each replay.

Each stored clip is tested unchanged, at quarter and sixteenth amplitude, and
at original/quarter amplitude with deterministic white noise (RMS 0.0045).
This is a stress screen, not a simulation of distance or an estimate of room
performance. No-segment, multiple-segment and duration-cap counts identify clips
for later listening and boundary annotation. An emitted segment is not proof
of correct or complete speech capture. Report processing time separately from
the 300 ms endpoint wait and the detector's internal hangover.

Silence, white noise and a 120 Hz tone are separate 30 second sanity checks.
Zero segments on these short artificial controls does not establish a real
false-wake or false-action rate. Unrelated speech is correctly speech to a VAD;
it must be rejected by downstream recognition.

## Next gate

Before enabling a candidate in the listener, replay fresh annotated continuous
room recordings through segmentation and the frozen wake/command recognizers.
Compare retained word boundaries, complete-request success, false wakes and
wrong intents at fixed thresholds. Evaluate the energy-based final trimming as
a separate change, since it may still remove weak speech. Include detector
hangover, queue delays and playback in true latency measurements. Do not select
a production default from this feasibility screen alone.
