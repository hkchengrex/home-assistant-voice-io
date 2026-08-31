# Benchmark

This page records recognition timings from a first-generation Microsoft Surface Go.

For remote voice generation, see the separate [OmniVoice batch benchmark](response-generation.md#batch-benchmark).

## Result

| | |
| --- | --- |
| Machine | First-generation Microsoft Surface Go |
| CPU | Intel Pentium Gold 4415Y @ 1.60 GHz |
| CPU layout | 2 cores / 4 threads |
| Runtime | Python 3.12 with the native matcher |
| Templates | 123 |
| Input phrase | 0.720 seconds |
| Measured runs | 100 after 10 warm-ups |
| Median | **86.862 ms** |
| p95 | **91.863 ms** |
| Mean | 88.266 ms |
| Range | 84.600–129.362 ms |

## What was measured

Each run processed one stored command from silence trimming through PCEN/MFCC feature extraction, dynamic-time-warping comparison, and final command classification. The accepted result was `lights_on`.

The test excludes microphone capture time, response playback, and external automation latency. In other words, it measures recognition after the utterance is available—not the time a person spends speaking.

## Live observation

In the same running deployment, the two most recent successful commands reached the Home Assistant action 150–190 ms after the detected command ended, with a 170 ms median. This is a small operational observation, not a controlled end-to-end benchmark.

## Read results carefully

Latency changes with phrase duration, number of templates, native versus NumPy matching, system load, and the speed of the connected automation. Compare systems with the same recordings and template library when evaluating changes.
