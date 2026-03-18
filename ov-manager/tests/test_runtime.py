"""Tests for ov_manager.runtime — XDG runtime directory resolution.

Chicago style: real filesystem via tmp_path.
"""

from __future__ import annotations

import importlib
import os

import pytest

# Module under test — will be created next
runtime_mod = importlib.import_module("ov_manager.runtime")


# ---------------------------------------------------------------------------
# XDG_RUNTIME_DIR
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_xdg_runtime_dir_used(monkeypatch, tmp_path):
    """When XDG_RUNTIME_DIR is set, uses <XDG>/ov-manager/."""
    xdg_dir = tmp_path / "xdg-runtime"
    xdg_dir.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))

    result = runtime_mod.get_runtime_dir()

    assert result == xdg_dir / "ov-manager"
    assert result.exists()


@pytest.mark.unit
def test_fallback_when_xdg_unset(monkeypatch, tmp_path):
    """When XDG_RUNTIME_DIR is unset, falls back to a temp-based dir."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    # Use a controlled tmpdir to avoid creating dirs in real /tmp
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    result = runtime_mod.get_runtime_dir()

    uid = os.getuid()
    expected_parent = tmp_path / f"runtime-{uid}" / "ov-manager"
    assert result == expected_parent
    assert result.exists()


@pytest.mark.unit
def test_directory_created_with_0700(monkeypatch, tmp_path):
    """Runtime directory is created with mode 0700."""
    xdg_dir = tmp_path / "xdg-runtime"
    xdg_dir.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))

    result = runtime_mod.get_runtime_dir()

    mode = result.stat().st_mode & 0o777
    assert mode == 0o700


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pidfile_write_read_cleanup(monkeypatch, tmp_path):
    """Write PID, read it back, then clean up."""
    xdg_dir = tmp_path / "xdg-runtime"
    xdg_dir.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))

    runtime_dir = runtime_mod.get_runtime_dir()
    pid_path = runtime_dir / "ovmgr-server.pid"

    # Write
    runtime_mod.write_pidfile(pid_path, 12345)
    assert pid_path.exists()
    assert pid_path.read_text().strip() == "12345"

    # Read
    assert runtime_mod.read_pidfile(pid_path) == 12345

    # Cleanup
    runtime_mod.remove_pidfile(pid_path)
    assert not pid_path.exists()


@pytest.mark.unit
def test_read_pidfile_returns_none_if_missing(tmp_path):
    """read_pidfile returns None when pidfile doesn't exist."""
    pid_path = tmp_path / "nonexistent.pid"
    assert runtime_mod.read_pidfile(pid_path) is None


@pytest.mark.unit
def test_remove_pidfile_noop_if_missing(tmp_path):
    """remove_pidfile does nothing when pidfile doesn't exist."""
    pid_path = tmp_path / "nonexistent.pid"
    runtime_mod.remove_pidfile(pid_path)  # Should not raise
