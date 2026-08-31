# Contributing

Install the development extra and run `pytest` before opening a change. New
platform-dependent behavior must include a portable fallback or a clear runtime
capability check. Tests must generate audio signals in memory; do not contribute
recordings of identifiable people.

For documentation changes, install the `docs` extra and run `mkdocs build --strict`.
Commit both the sources in `documentation/` and the generated site in `docs/`.
The sitemap override omits build-date modification claims, and the documentation
hook fixes the gzip timestamp so local and CI builds match across dates.
