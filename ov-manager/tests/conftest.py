"""Shared fixtures for ov-manager tests."""

from __future__ import annotations

import os

import pytest


@pytest.fixture()
def clean_env(monkeypatch):
    """Remove all OVMGR_* and HF_TOKEN env vars to ensure test isolation."""
    for key in list(os.environ):
        if key.startswith("OVMGR_") or key == "HF_TOKEN":
            monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def models_dir(tmp_path):
    """Return a temporary models directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture()
def config_json(models_dir):
    """Return the path to config.json inside the temp models dir."""
    return models_dir / "config.json"


@pytest.fixture()
def toml_dir(tmp_path):
    """Return a temp directory suitable for writing ov-manager.toml into."""
    d = tmp_path / "project"
    d.mkdir()
    # Create a .git marker so _find_toml stops here
    (d / ".git").mkdir()
    return d
