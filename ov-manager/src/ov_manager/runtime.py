"""XDG-compliant runtime directory and PID file helpers.

Follows the XDG Base Directory Specification for ``XDG_RUNTIME_DIR``.
PID files are ephemeral runtime artifacts and belong in the runtime dir.

When ``XDG_RUNTIME_DIR`` is not set (common on non-systemd systems), falls
back to ``<tempdir>/runtime-<uid>/ov-manager/`` with a warning.
"""

from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path

APP_NAME = "ov-manager"


def get_runtime_dir() -> Path:
    """Resolve the XDG runtime directory for ov-manager.

    Search order:

    1. ``XDG_RUNTIME_DIR`` → ``<XDG_RUNTIME_DIR>/ov-manager/``
    2. Fallback → ``<tempdir>/runtime-<uid>/ov-manager/``

    The directory is created with mode ``0700`` if it does not exist.

    Returns:
        Path to the ov-manager runtime directory.
    """
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")

    if runtime_dir:
        base = Path(runtime_dir)
    else:
        base = Path(tempfile.gettempdir()) / f"runtime-{os.getuid()}"
        base.mkdir(mode=0o700, exist_ok=True)
        warnings.warn(
            f"XDG_RUNTIME_DIR is not set; falling back to {base}",
            RuntimeWarning,
            stacklevel=2,
        )

    app_dir = base / APP_NAME
    app_dir.mkdir(mode=0o700, exist_ok=True)
    return app_dir


def write_pidfile(path: Path, pid: int) -> None:
    """Write a PID to a file.

    Args:
        path: Path to the pidfile.
        pid: Process ID to write.
    """
    path.write_text(str(pid))


def read_pidfile(path: Path) -> int | None:
    """Read a PID from a file.

    Args:
        path: Path to the pidfile.

    Returns:
        The PID as an integer, or ``None`` if the file does not exist or is
        not a valid integer.
    """
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except (ValueError, OSError):
        return None


def remove_pidfile(path: Path) -> None:
    """Remove a pidfile if it exists.

    Args:
        path: Path to the pidfile.
    """
    path.unlink(missing_ok=True)
