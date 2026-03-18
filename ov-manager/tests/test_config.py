"""Tests for ov_manager.config — config resolution with server/webui fields.

Chicago style: real objects, real filesystem via tmp_path.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

config_mod = importlib.import_module("ov_manager.config")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_defaults_when_no_sources(clean_env, tmp_path):
    """All server/webui fields get hardcoded defaults when no config sources exist."""
    cfg = config_mod.resolve(cwd=tmp_path)

    assert cfg.server_host == "127.0.0.1"
    assert cfg.server_port == 8100
    assert cfg.webui_host == "127.0.0.1"
    assert cfg.webui_port == 3100
    assert cfg.webui_data_dir is None
    assert cfg.webui_image == "ghcr.io/open-webui/open-webui:main"


@pytest.mark.unit
def test_webui_port_default_3100(clean_env, tmp_path):
    """Verify the webui port default is 3100, not 3000."""
    cfg = config_mod.resolve(cwd=tmp_path)
    assert cfg.webui_port == 3100


# ---------------------------------------------------------------------------
# TOML overrides defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_toml_overrides_defaults(clean_env, toml_dir):
    """Values in ov-manager.toml override hardcoded defaults."""
    toml_file = toml_dir / "ov-manager.toml"
    toml_file.write_text(
        'server_host = "0.0.0.0"\n'
        "server_port = 9000\n"
        'webui_host = "0.0.0.0"\n'
        "webui_port = 4000\n"
        'webui_data_dir = "/data/webui"\n'
        'webui_image = "ghcr.io/open-webui/open-webui:cuda"\n'
    )

    cfg = config_mod.resolve(cwd=toml_dir)

    assert cfg.server_host == "0.0.0.0"
    assert cfg.server_port == 9000
    assert cfg.webui_host == "0.0.0.0"
    assert cfg.webui_port == 4000
    assert cfg.webui_data_dir == Path("/data/webui").resolve()
    assert cfg.webui_image == "ghcr.io/open-webui/open-webui:cuda"


# ---------------------------------------------------------------------------
# Env vars override TOML
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_env_overrides_toml(clean_env, monkeypatch, toml_dir):
    """OVMGR_* env vars override values from ov-manager.toml."""
    toml_file = toml_dir / "ov-manager.toml"
    toml_file.write_text('server_port = 9000\nwebui_image = "ghcr.io/open-webui/open-webui:cuda"\n')

    monkeypatch.setenv("OVMGR_SERVER_PORT", "7777")
    monkeypatch.setenv("OVMGR_WEBUI_IMAGE", "ghcr.io/open-webui/open-webui:dev")

    cfg = config_mod.resolve(cwd=toml_dir)

    assert cfg.server_port == 7777
    assert cfg.webui_image == "ghcr.io/open-webui/open-webui:dev"


@pytest.mark.unit
def test_env_server_host(clean_env, monkeypatch, tmp_path):
    """OVMGR_SERVER_HOST env var is picked up."""
    monkeypatch.setenv("OVMGR_SERVER_HOST", "0.0.0.0")
    cfg = config_mod.resolve(cwd=tmp_path)
    assert cfg.server_host == "0.0.0.0"


@pytest.mark.unit
def test_env_webui_host(clean_env, monkeypatch, tmp_path):
    """OVMGR_WEBUI_HOST env var is picked up."""
    monkeypatch.setenv("OVMGR_WEBUI_HOST", "192.168.1.10")
    cfg = config_mod.resolve(cwd=tmp_path)
    assert cfg.webui_host == "192.168.1.10"


@pytest.mark.unit
def test_env_webui_port(clean_env, monkeypatch, tmp_path):
    """OVMGR_WEBUI_PORT env var is picked up."""
    monkeypatch.setenv("OVMGR_WEBUI_PORT", "5555")
    cfg = config_mod.resolve(cwd=tmp_path)
    assert cfg.webui_port == 5555


@pytest.mark.unit
def test_env_webui_data_dir(clean_env, monkeypatch, tmp_path):
    """OVMGR_WEBUI_DATA_DIR env var is picked up."""
    monkeypatch.setenv("OVMGR_WEBUI_DATA_DIR", "/srv/webui-data")
    cfg = config_mod.resolve(cwd=tmp_path)
    assert cfg.webui_data_dir == Path("/srv/webui-data").resolve()


# ---------------------------------------------------------------------------
# CLI overrides env
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_overrides_env(clean_env, monkeypatch, tmp_path):
    """CLI keyword arguments override env vars."""
    monkeypatch.setenv("OVMGR_SERVER_HOST", "10.0.0.1")
    monkeypatch.setenv("OVMGR_SERVER_PORT", "9999")

    cfg = config_mod.resolve(
        server_host="192.168.0.1",
        server_port=1234,
        cwd=tmp_path,
    )

    assert cfg.server_host == "192.168.0.1"
    assert cfg.server_port == 1234


@pytest.mark.unit
def test_cli_webui_overrides(clean_env, tmp_path):
    """CLI webui keyword arguments are applied."""
    cfg = config_mod.resolve(
        webui_host="0.0.0.0",
        webui_port=8888,
        webui_data_dir=Path("/my/data"),
        webui_image="custom:latest",
        cwd=tmp_path,
    )

    assert cfg.webui_host == "0.0.0.0"
    assert cfg.webui_port == 8888
    assert cfg.webui_data_dir == Path("/my/data").resolve()
    assert cfg.webui_image == "custom:latest"


# ---------------------------------------------------------------------------
# _find_toml search behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_find_toml_walks_to_git_root(tmp_path):
    """_find_toml walks up from cwd to find ov-manager.toml at git root."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "ov-manager.toml").write_text('backend = "docker"\n')

    subdir = root / "deep" / "nested"
    subdir.mkdir(parents=True)

    found = config_mod._find_toml(subdir)
    assert found == root / "ov-manager.toml"


@pytest.mark.unit
def test_find_toml_stops_at_git_root(tmp_path):
    """TOML above the .git root is not found."""
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "ov-manager.toml").write_text('backend = "docker"\n')

    repo = parent / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    found = config_mod._find_toml(repo)
    assert found is None


@pytest.mark.unit
def test_find_toml_ovmgr_config_env(clean_env, monkeypatch, tmp_path):
    """OVMGR_CONFIG env var points to an explicit config file."""
    cfg_file = tmp_path / "custom.toml"
    cfg_file.write_text('backend = "baremetal"\n')

    monkeypatch.setenv("OVMGR_CONFIG", str(cfg_file))

    found = config_mod._find_toml(tmp_path)
    assert found == cfg_file


@pytest.mark.unit
def test_find_toml_ovmgr_config_missing_errors(clean_env, monkeypatch, tmp_path):
    """OVMGR_CONFIG pointing to a nonexistent file raises an error."""
    monkeypatch.setenv("OVMGR_CONFIG", str(tmp_path / "nonexistent.toml"))

    with pytest.raises(Exception, match="non-existent"):
        config_mod._find_toml(tmp_path)


@pytest.mark.unit
def test_find_toml_user_level_fallback(clean_env, monkeypatch, tmp_path):
    """Falls back to ~/.config/ov-manager/ov-manager.toml."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))

    user_cfg_dir = fake_home / ".config" / "ov-manager"
    user_cfg_dir.mkdir(parents=True)
    user_cfg = user_cfg_dir / "ov-manager.toml"
    user_cfg.write_text('backend = "docker"\n')

    # cwd has no toml and no .git — so it walks to fs root and finds nothing
    # then falls back to user config
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Patch Path.home() to use our fake home
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

    found = config_mod._find_toml(empty_dir)
    assert found == user_cfg


# ---------------------------------------------------------------------------
# Integer resolution for port fields
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_toml_port_as_integer(clean_env, toml_dir):
    """TOML integer values for ports resolve correctly."""
    toml_file = toml_dir / "ov-manager.toml"
    toml_file.write_text("server_port = 5000\nwebui_port = 6000\n")

    cfg = config_mod.resolve(cwd=toml_dir)
    assert cfg.server_port == 5000
    assert cfg.webui_port == 6000


# ---------------------------------------------------------------------------
# Auto backend resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_auto_resolves_to_baremetal_when_ovms_on_path(clean_env, monkeypatch, tmp_path):
    """backend='auto' resolves to 'baremetal' when ovms is on $PATH."""

    def fake_which(name):
        if name == "ovms":
            return "/usr/bin/ovms"
        return None

    monkeypatch.setattr("shutil.which", fake_which)

    cfg = config_mod.resolve(cwd=tmp_path)
    assert cfg.backend == "baremetal"


@pytest.mark.unit
def test_auto_resolves_to_docker_when_only_docker_on_path(clean_env, monkeypatch, tmp_path):
    """backend='auto' resolves to 'docker' when only docker is on $PATH."""

    def fake_which(name):
        if name == "docker":
            return "/usr/bin/docker"
        return None

    monkeypatch.setattr("shutil.which", fake_which)

    cfg = config_mod.resolve(cwd=tmp_path)
    assert cfg.backend == "docker"


@pytest.mark.unit
def test_auto_errors_when_neither_found(clean_env, monkeypatch, tmp_path):
    """backend='auto' raises an error when neither ovms nor docker is found."""
    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(Exception, match="(?i)no.*backend|not found"):
        config_mod.resolve(cwd=tmp_path)


@pytest.mark.unit
def test_explicit_backend_not_resolved(clean_env, monkeypatch, tmp_path):
    """Explicit backend='docker' is kept as-is, not auto-resolved."""
    # Even if ovms is on $PATH, explicit docker stays docker
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ovms" if name == "ovms" else None)

    cfg = config_mod.resolve(backend="docker", cwd=tmp_path)
    assert cfg.backend == "docker"
