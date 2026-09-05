# Full-bank feature and matching experiment

`ha_voice.bank_experiment` compares MFCC and PCEN representations and template
aggregation on an entire labeled recording bank. It is offline and does not
invoke a listener, playback, or actions.

```bash
python -m ha_voice.bank_experiment \
  --recordings /path/to/recordings --config /path/to/commands.toml \
  --archive backup --output /path/to/private-results
```

Use an environment containing the compiled native DTW matcher. All evaluation
artifacts contain recording-linked data and belong in private storage, not the
public repository. The evaluator writes a complete inventory, decoded-audio
hashes, distance matrices, per-profile reports, and a combined report.

## Current evaluation: repeated random holdouts

Use `ha_voice.bank_random` to evaluate the saved distance matrices with random
holdouts. Recording dates are not used as environment or session labels.

```bash
python -m ha_voice.bank_random --recordings /path/to/recordings \
  --config /path/to/commands.toml --archive backup --results /path/to/private-results
```

This evaluates all five feature profiles, both cosine profiles, and both fusion
profiles using three repeats of stratified random five-fold holdout. Seeds are
20260905, 20260906 and 20260907. Splits are stratified by recording label; every
recording is held out once per repeat. All variants use the same saved splits
within a corpus scope. Exact duplicate decoded audio has already been removed,
and the splitter rejects duplicate audio identities.

Current-only and combined-bank scopes are reported separately. Thresholds use
only training-set leave-one-out decisions, preserving the earlier fitting rule
so the outer split method is the changed variable. No test labels tune thresholds.
The vectorized scorer is tested against the original scalar scorer, including
self exclusion and equivalent-intent aggregation.

`random-holdout.json` contains results and per-repeat metrics;
`random-splits.json` preserves the exact audio identities in every split.
Aggregate counts are repeated test observations, not distinct recordings. Do not
interpret their sum as independent samples when estimating uncertainty. Existing
hardware latency measurements still apply because the features and distance
computations have not changed.

The date-based results described below are retained for historical comparison;
they are not the primary evaluation protocol.

## Corpus and original date-based folds

The tool inventories every WAV recursively. It scores direct labeled folders
and labeled folders one level beneath explicitly named archives. Accepted labels
are configured commands, the configured wake/calibration labels, `_not_command`,
and `_not_start_phrase`. Unknown, retired, and withdrawn folders remain visible
in the manifest but receive no guessed labels. Audit these before widening the
scored corpus. Equal decoded audio is deduplicated; conflicting labels are
excluded. This does not detect every near-duplicate or recut recording.

A filename's initial YYYYMMDD defines its recording-day fold. Holding out whole
days is a conservative session proxy; it is not verified session or speaker
metadata. Unknown dates participate only in leave-one-audio-out ranking. A
held-out positive whose intent has no remaining template is counted separately
as unsupported, not silently counted as a recognized or ordinary failed class.

Wake classification treats the wake as positive and other labeled speech as
negative. Command classification uses configured action groups, not wording
aliases as competing actions. Known wake clips and calibration clips are command
rejection queries, but are not added to command rejection templates. A clip
labeled only not-wake is excluded from command scoring because it might contain
a valid command.

## Controlled comparisons

Capture and trimming behavior stay fixed. There are five feature profiles:

- Existing MFCC: 13 cepstra plus 13 deltas.
- Existing PCEN cepstra: 13 cepstra plus 13 deltas.
- MFCC with a per-utterance variance floor of one quarter of median coefficient
  variance, to limit amplification of low-variance coefficients.
- PCEN with that same variance floor.
- PCEN with the floor and 24 cepstra plus 24 deltas, to test retained detail.

These are explicit hypotheses, not established improvements. PCEN parameters
remain alpha 0.98, delta 2, exponent 0.5, and smoothing 0.05. FFT, mel filters,
windows and hop sizes match the corresponding current frontend. The floor
requires no statistics fitted on held-out examples.

Each profile is scored with nearest-template aggregation k=1, 2 and 3. All
eligible templates are compared using the existing DTW recurrence; there is no
approximate shortlist. Therefore the baseline feature profile is a full-search
reference, not an exact reproduction of production decisions. Both directions
of every pair are computed rather than assuming tie handling is symmetric.

## Reading results

Leave-one-audio-out results are ranking diagnostics. Their acceptance fields
are unused, because no acceptance threshold is fitted in that branch. Adjacent
recordings can remain in the training pool, so these results can be optimistic.

For each held-out day, distance/margin thresholds are chosen using only
leave-one-audio-out predictions from the other days. The finite search maximizes
correct training accepts subject to zero observed training wrong accepts. It
uses 31 distance quantiles and margins 0, .01, .02, .03, .04, .06, .08, .12, .2,
.3, .5. This deliberately conservative rule may reject many correct queries;
zero training errors do not guarantee zero held-out errors. Wrong accepts
include both non-requests accepted and requests assigned to the wrong intent.

Held-out recall is correct accepts / represented positive queries. Precision is
correct accepts / all accepts; a profile that accepts nothing has undefined
precision. Its value depends on the evaluation bank's class balance and is not
a household false-wake rate. Inspect per-day results and unsupported counts as
well as aggregate metrics.

Using these results to pick a feature profile makes the bank development data.
Validate the frozen choice on a new recording session before deployment. Saved
clip matching alone cannot measure complete-request recall. Extraction timings
are measured on the execution host; the all-pairs run time is not per-request
latency. Benchmark final candidate matching separately on the target hardware.

## Follow-up comparisons

With the same bank and saved matrices, run:

```bash
python -m ha_voice.bank_followup --recordings /path/to/recordings \
  --config /path/to/commands.toml --archive backup --results /path/to/private-results
python -m ha_voice.bank_metric --recordings /path/to/recordings \
  --config /path/to/commands.toml --archive backup --results /path/to/private-results
```

The follow-up compares the current folders separately from the combined archive
bank and evaluates 50/50 and 75/25 MFCC/PCEN distance fusion. Fusion means a
weighted average of two DTW distances, not concatenated features. It adds a
second match computation at inference; use it only if its accuracy benefit
justifies that cost.

The metric experiment replaces Euclidean frame costs with cosine costs from
unit-normalized frames. It preserves the native DTW recurrence, band, and path
normalization. A zero feature vector has cosine cost one, even against another
zero vector; this is not a silence detector. The original score thresholds must
not be reused unchanged. Both scripts verify the ordered audio identities before
using cached matrices. Preserve the original configuration and source versions
alongside the results; matching labels/settings must also remain fixed.

Latency tests use eight positive recordings spanning the duration range, each
repeated three times, with precomputed template features. They include query
features and matching, but exclude capture, trimming, actions and playback.
Euclidean timing reports full-search and production-style eight-per-folder
shortlisting separately. Cosine timing uses full search. These bank benchmarks
are not measurements of the exact deployed listener or independent accuracy
samples. Evaluate precision and latency separately: a fast incorrect match is
not an improvement.
