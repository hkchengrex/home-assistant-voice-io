# OmniVoice Space batch extension

This folder backs up the customized entry point and batch API for
`hkchengrex/OmniVoice`. The model and original demo remain unchanged.
The original Space revision is `64deef91cf1f7bfdfc97ff07502b4cd47a0e361e`.

To rebuild, duplicate that revision of the Space (including its `omnivoice/`
sources, `requirements.txt`, and README metadata), then upload `app.py` and
`batch_api.py` from this folder to its root. Keep the existing ZeroGPU hardware.
Use a local Hugging Face login with write access; never commit credentials or
household recordings. These two files are distributed under Apache-2.0, matching
the upstream Space from which the entry point is derived.

For an existing Space, install the library's `voice-clone` extra, inspect the
current Space commit, then run `python deploy.py --expected-revision COMMIT`.
The script uploads only the two runtime files and refuses to overwrite changes
made since inspection. Hugging Face rebuilds the Space after the upload.

`/_clone_fn` and the voice-design API remain available. `/clone_batch` takes the
same cloning controls, but its first argument is a JSON list of texts and its
last argument is GPU batch size (default 4, range 1–8). It returns PCM16 WAV files
in input order and a JSON timing report. One reference prompt is prepared per
request and reused in true model batches, not independent network requests.

Limits: 8 lines per request, 300 characters per line, 1200 characters total,
1–20 second reference, optional fixed duration at most 15 seconds per clip.
The GPU task has a 120-second ceiling. Split long requests or lower batch size
after an out-of-memory error. Validation happens before GPU allocation; all
generation endpoints share a one-job concurrency limit.

The report separates prompt preparation, generation, and total GPU-task time.
Client wall time additionally includes queueing, uploads, and downloads. File
cache cleanup runs hourly for files older than an hour. Voice prompts are reused
within a request only, never cached across users.
