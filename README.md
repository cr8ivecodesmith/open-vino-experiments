# OpenVINO Experiments

A workspace for OpenVINO tooling experiments and utilities.

## Projects

### `ov-testdata-scraper`

A CLI tool that mirrors the [OpenVINO test data bucket](https://storage.openvinotoolkit.org/data/test_data/)
to a local directory. Supports parallel downloads, random request delays, and
resume-on-interrupt. See [`ov-testdata-scraper/README.md`](ov-testdata-scraper/README.md) for full
documentation.

## Setup

Requires Python 3.12+ and [`uv`](https://github.com/astral-sh/uv).

```bash
uv pip install -r requirements.txt
```

This installs all sub-projects in editable mode, making their CLI entry points
available in the active virtual environment.
