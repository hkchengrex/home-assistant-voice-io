# PCEN parameter sweep

`python -m ha_voice.bank_pcen` evaluates 23 predeclared PCEN parameter settings
on a labeled recording bank. It is an offline command-recognition experiment;
it does not change listener settings or execute actions.

Build the native DTW extension (`python setup.py build_ext --inplace`) with
setuptools, a C compiler and Python development headers. Install NumPy and run:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ha_voice.bank_pcen \
  --recordings /path/to/recordings --config /path/to/commands.toml \
  --archive backup --output /path/to/private-results --workers 4 --top-k 2
```

Run again with `--top-k 3` and a different output directory to inspect that
matcher setting. Workers are ordinary local processes. Output inventories contain
recording paths, labels and hashes and belong with the private bank, not here.

The grid crosses smoothing {0.01, 0.025, 0.05, 0.1} with alpha
{0.8, 0.9, 0.95, 0.98}. Seven additional settings vary root {0.25, 0.75},
delta {0.5, 1, 4}, or epsilon {1e-8, 1e-4}, one at a time around the baseline.
The unchanged baseline is smoothing 0.05, alpha 0.98, root 0.5, delta 2,
epsilon 1e-6. This is a bounded search, not the full five-dimensional Cartesian
product; interactions between compression and adaptation remain untested.

`extract_pcen_cepstra` exposes alpha, delta, root and epsilon as keyword-only
arguments. Runtime `extract_command_features` continues using the old defaults.
The spectrum, 40 filters, 13 cepstra plus deltas, utterance normalization,
first-frame smoother initialization, trimming and Euclidean DTW remain fixed.
Every ordered pair is computed without a shortlist, preserving DTW tie handling.

## Evaluation

Exact duplicate decoded recordings are deduplicated. The active and complete
bank are evaluated separately, using three repeats of five stratified random
folds (seeds 20260905, 20260906, 20260907). Each candidate uses identical splits.
Dates are not interpreted as environments. Each recording contributes three
test observations; these are not independent recordings.

For every candidate, distance and margin thresholds are fitted using only
training-fold leave-one-out scores, requiring zero training wrong accepts and
maximizing training correct accepts. Outer test results are reported separately.
The descriptive PR envelope instead uses evaluation labels to select cutoffs;
its zero-error points measure observed separation, not validated thresholds.

An additional nested evaluation chooses a candidate separately inside each outer
training fold, using three inner stratified folds (seed 9173). It minimizes inner
wrong accepts first, then maximizes correct accepts, with grid-order ties favoring
the baseline. It then takes that candidate's training-calibrated outer predictions.
Outer labels are never used by this selector. Reporting the best candidate after
inspecting all outer results is still exploratory model selection.

Artifacts include module/config/audio hashes, inventories, exact split identities,
per-candidate fold results, descriptive PR operating points and nested selections.
Distance caches are ignored by Git and can be regenerated. Re-running with the
same provenance reuses the caches. Source/config/audio changes invalidate them.

This evaluates completed command clips, not streaming VAD, wake recognition or
end-to-end request success. Archive label uncertainty and reuse of the same bank
limit the strength of conclusions. No audio is sent to a third-party service.
