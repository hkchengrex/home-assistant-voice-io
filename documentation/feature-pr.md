# Offline feature precision-recall comparisons

These optional analysis tools compare feature representations and matching rules
on saved repeated random holdouts. They do not change the runtime recognizer.
Use Python 3.12 with the pinned optional packages:

```bash
python -m venv .venv-feature-pr
.venv-feature-pr/bin/python -m pip install . -r documentation/feature-pr-requirements.txt
```

On Windows, the interpreter is `.venv-feature-pr/Scripts/python.exe`.
First create the inventory and original distance caches with `bank_experiment`,
then run `bank_followup`, `bank_metric` and `bank_random` against the same bank and
configuration. See those modules' `--help` for input arguments. Keep recordings,
configuration, split manifests and raw predictions private.

## PNCC candidate

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m ha_voice.bank_pncc \
  --recordings /private/recordings --config /private/commands.toml \
  --archive backup --results /private/results
```

The results directory must already contain the original `manifest.json`.
The evaluator rejects a changed ordered audio inventory. It uses the same
random split function, full Euclidean DTW, nearest-template aggregation and
intent grouping as the earlier bank experiment. There is no learned transform
or test-fitted feature normalization. Utterance normalization uses only that
utterance, as in the baseline.

This PNCC variant composes SPAFE 0.3.3 components: 40 gammatone channels,
medium-time power estimation, asymmetric noise suppression, temporal masking,
mean-power normalization, power-law compression, and 13 DCT coefficients.
It uses the existing 16 kHz framing (25 ms / 10 ms hop, 512-point FFT), Hamming
window, pre-emphasis .97, per-coefficient utterance mean/variance normalization,
and 13 delta coefficients. The result has 26 dimensions like the baseline.
SPAFE's weight ratio receives a denominator floor of
`max(max(medium_power) * 1e-12, 1e-30)` to handle silent channels before division.
It is an explicitly specified PNCC variant, not claimed to be a bitwise match
to the paper's implementation.

Background: [Kim and Stern's PNCC paper](https://research.google/pubs/power-normalized-cepstral-coefficients-pncc-for-robust-speech-recognition/).
The optional implementation components come from
[SPAFE](https://superkogito.github.io/spafe/features/pncc.html), distributed under
the BSD 3-Clause license. SPAFE and SciPy remain offline dependencies.

Repeat the same command with `--timing-only` to benchmark all three frontends
on eight duration-stratified clips, with one warm-up and three measured passes.
Set the BLAS/OpenMP thread environment consistently. Timing excludes matching,
audio buffering, VAD and endpoint delay.

## Curves

```bash
python -m ha_voice.bank_pr \
  --input /private/results/random-holdout.json \
  --input /private/results/pncc.json \
  --output /private/aggregate-pr
```

The plotter verifies matching query identities, targets and folds across all
profiles within a scope/head. Feature plots hold Euclidean DTW and top-k fixed
(k=2 for commands, k=3 for wake). Separate matching plots label cosine and
fusion curves explicitly. MFCC is the wake baseline; PCEN is the command baseline.

Acceptance is `distance <= d` and signed competitor `margin >= m`, with `m >= 0`.
A wrong intent and an accepted negative both count as false accepts. Recall is
correct accepts divided by all positive observations. Unsupported or non-finite
positive queries remain in the denominator; unsupported labels are never used
as an oracle rejection rule. Only finite scores can be accepted.

The plotter enumerates distinct observed distance/margin thresholds through
2D cumulative counts, keeps the minimum false-accept count for each true-accept
count, then removes dominated points. The empty acceptance set is displayed at
(0, 1) by convention. Lines connect observed points as guides; they do not imply
that fractional intermediate operating points are available. Raw curves are
not standard single-score PR curves, so no average-precision score is reported.

**These are exploratory threshold envelopes.** Evaluation labels select the
frontier and must not be used to claim independent validation of a chosen
threshold. Random holdouts keep test clips out of the template bank, but repeated
observations of the same clip are not independent samples. Corpus precision
depends on the sampled positive/negative prevalence. Full-search curves are not
the live shortlisting path, and do not measure household false alarms per hour.
Use a separate random calibration split inside each training fold when validating
a deployment candidate. No recording-day grouping is required.

Outputs include high-precision zooms and full-range plots (PNG/SVG/PDF), a CSV
of nondominated-by-TP operating points, and aggregate JSON with per-repeat views.
They omit audio identities and raw predictions. The JSON also includes requested
wrong-accept budgets and precision levels, which need not coincide with every
point retained in the plotted Pareto frontier.

Validation:

```bash
python -m pytest tests/test_bank_pr.py tests/test_bank_pncc.py
```

The PNCC tests skip when optional packages are absent; run them explicitly in
the optional environment. They cover silent channels, very quiet synthetic
signals and invalid input. PR tests compare the cumulative method to brute-force
threshold enumeration and check wrong-intent accounting and recall denominators.
