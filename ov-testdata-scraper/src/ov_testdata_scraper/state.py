"""Persistent download state management.

Tracks which files have been successfully downloaded so that interrupted runs
can be resumed without re-downloading completed files.  State is persisted as a
JSON file at ``<out_dir>/.scraper_state.json``.

All writes go through a temporary file + atomic rename to prevent corruption if
the process is killed mid-write.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

STATE_FILENAME = ".scraper_state.json"


def _state_path(out_dir: Path) -> Path:
    """Return the path to the state file inside *out_dir*."""
    return out_dir / STATE_FILENAME


def load(out_dir: Path) -> dict[str, dict]:
    """Load the persisted state dict from *out_dir*.

    Returns an empty dict when the state file does not exist or is corrupt.
    """
    path = _state_path(out_dir)
    if not path.exists():
        return {}
    try:
        return cast(dict[str, dict], json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return {}


def save(out_dir: Path, state: dict[str, dict]) -> None:
    """Atomically persist *state* to ``<out_dir>/.scraper_state.json``.

    Writes to a temporary file in the same directory first, then renames so
    that a crash during the write never leaves a half-written state file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = _state_path(out_dir)
    # Write to a tmp file in the same directory so os.replace is atomic.
    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".tmp", prefix=".scraper_state_")
    try:
        with open(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        Path(tmp_path).replace(dest)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise


def clear(out_dir: Path) -> None:
    """Remove the state file from *out_dir* if it exists."""
    path = _state_path(out_dir)
    path.unlink(missing_ok=True)


def record_download(state: dict[str, dict], rel_path: str, size: int) -> None:
    """Record a successful download in *state* (in-memory only; call :func:`save` to persist)."""
    state[rel_path] = {
        "size": size,
        "downloaded_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def is_downloaded(state: dict[str, dict], rel_path: str, expected_size: int) -> bool:
    """Return ``True`` if *rel_path* is recorded in *state* with a matching size."""
    entry = state.get(rel_path)
    if entry is None:
        return False
    return entry.get("size") == expected_size
