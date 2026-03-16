# ov-testdata-scraper

A CLI tool that mirrors the OpenVINO test data bucket at
`https://storage.openvinotoolkit.org/data/test_data/` to a local directory.

## Features

- **Single manifest fetch** — downloads `filetree.json` once to enumerate all
  files; no recursive HTML crawling
- **Parallel downloads** — configurable worker thread count
- **Per-thread random delay** — polite, randomised delay before each HTTP
  request
- **Resume on interrupt** — atomic state file tracks completed downloads so
  re-runs skip already-downloaded files
- **Dry-run mode** — list pending files and total size without downloading
- **Refresh mode** — re-download everything from scratch (with confirmation)

## Installation

From the repository root:

```bash
uv pip install -r requirements.txt
```

Or install this package directly:

```bash
uv pip install -e ./ov-testdata-scraper
```

## Usage

```
ov-testdata-scraper [OPTIONS]

Options:
  -o, --out-dir PATH     Directory to mirror downloaded test data into.
                         [default: ov-testdata]
  -w, --workers INTEGER  Number of parallel download threads.  [default: 4]
  -d, --delay FLOAT      Maximum random delay (seconds) per-thread before each
                         HTTP request.  [default: 2.0]
      --dry-run          List files that would be downloaded without
                         downloading.
      --refresh          Re-download all files, ignoring previous progress.
                         Prompts for confirmation.
  -h, --help             Show this message and exit.
```

### Examples

```bash
# Download everything to ./ov-testdata (default)
ov-testdata-scraper

# Custom output directory
ov-testdata-scraper --out-dir /data/ov-testdata

# Preview what would be downloaded
ov-testdata-scraper --dry-run

# Faster downloads with more threads and no delay
ov-testdata-scraper --workers 8 --delay 0

# Re-download everything
ov-testdata-scraper --refresh
```

## Resume behaviour

Progress is tracked in `<out-dir>/.scraper_state.json`. Each successfully
downloaded file is recorded with its remote size and a timestamp. On
subsequent runs:

- Files whose local size matches the remote manifest size are skipped.
- Files that are missing or whose size differs are (re-)downloaded.
- `--refresh` deletes the state file (after confirmation) and re-downloads
  all files.

Killing the process mid-run is safe — only completed downloads are recorded,
so the next run picks up exactly where the interrupted one left off.

## Project structure

```
ov-testdata-scraper/
├── pyproject.toml
└── src/
    └── ov_testdata_scraper/
        ├── __init__.py
        ├── cli.py        # Click entry point
        ├── scraper.py    # Manifest fetch, tree walk, parallel download
        └── state.py      # Atomic JSON state persistence
```

## Development

Install with dev dependencies:

```bash
uv pip install -e "./ov-testdata-scraper[dev]"
```

Run linting and type checks:

```bash
ruff check src/
mypy src/
```
