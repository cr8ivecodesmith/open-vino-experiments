"""Configuration resolution for ov-manager.

Priority order (highest to lowest):
    1. CLI flags (passed in explicitly as non-None values)
    2. ``OVMGR_*`` environment variables
    3. ``ov-manager.toml`` in the current working directory
    4. Hardcoded defaults

HuggingFace token resolution:
    ``--token`` CLI flag → ``OVMGR_HF_TOKEN`` → ``HF_TOKEN`` → None
"""

from __future__ import annotations

import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import click

TOML_FILENAME = "ov-manager.toml"

VALID_BACKENDS = ("auto", "baremetal", "docker")
DEFAULT_BACKEND = "auto"
DEFAULT_MODELS_DIR = Path.home() / ".models"
DEFAULT_CACHE_DIR = Path.home() / ".models" / "cache"
DEFAULT_DOCKER_IMAGE = "openvino/model_server:latest"

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8100
DEFAULT_WEBUI_HOST = "127.0.0.1"
DEFAULT_WEBUI_PORT = 3100
DEFAULT_WEBUI_IMAGE = "ghcr.io/open-webui/open-webui:main"


@dataclass
class Config:
    """Resolved configuration for a single ov-manage invocation."""

    backend: str = DEFAULT_BACKEND
    models_dir: Path = field(default_factory=lambda: DEFAULT_MODELS_DIR)
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    hf_token: str | None = None
    docker_image: str = DEFAULT_DOCKER_IMAGE

    # Server
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT

    # Open WebUI
    webui_host: str = DEFAULT_WEBUI_HOST
    webui_port: int = DEFAULT_WEBUI_PORT
    webui_data_dir: Path | None = None
    webui_image: str = DEFAULT_WEBUI_IMAGE

    @property
    def config_json_path(self) -> Path:
        """Return the path to the OVMS top-level config.json."""
        return self.models_dir / "config.json"


def _find_toml(cwd: Path | None = None) -> Path | None:
    """Locate the ``ov-manager.toml`` config file.

    Search order:

    1. ``OVMGR_CONFIG`` environment variable — explicit path to a config file.
    2. Walk up from *cwd* toward the filesystem root, stopping at a ``.git``
       directory (project root convention). Returns the first ``ov-manager.toml``
       found.
    3. ``~/.config/ov-manager/ov-manager.toml`` — user-level fallback.

    Args:
        cwd: Starting directory for the upward walk. Defaults to ``Path.cwd()``.

    Returns:
        Path to the config file, or ``None`` if not found.

    Raises:
        click.ClickException: If ``OVMGR_CONFIG`` is set but points to a
            non-existent file.
    """
    # 1. Explicit env var
    env_path = os.environ.get("OVMGR_CONFIG")
    if env_path:
        p = Path(env_path)
        if not p.exists():
            raise click.ClickException(f"OVMGR_CONFIG points to a non-existent file: {p}")
        return p

    # 2. Walk up from cwd, stop at .git or filesystem root
    search = (cwd or Path.cwd()).resolve()
    while True:
        candidate = search / TOML_FILENAME
        if candidate.exists():
            return candidate
        # Stop at git root or filesystem root
        if (search / ".git").exists() or search.parent == search:
            break
        search = search.parent

    # 3. User-level config
    user_config = Path.home() / ".config" / "ov-manager" / TOML_FILENAME
    if user_config.exists():
        return user_config

    return None


def _load_toml(cwd: Path | None = None) -> dict:
    """Load ``ov-manager.toml`` using :func:`_find_toml` search order.

    Args:
        cwd: Starting directory for the upward walk. Defaults to ``Path.cwd()``.

    Returns:
        Parsed TOML as a dict, or empty dict if no config file is found.
    """
    toml_path = _find_toml(cwd)
    if toml_path is None:
        return {}
    with toml_path.open("rb") as fh:
        return tomllib.load(fh)


def resolve(
    *,
    backend: str | None = None,
    models_dir: Path | None = None,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
    docker_image: str | None = None,
    server_host: str | None = None,
    server_port: int | None = None,
    webui_host: str | None = None,
    webui_port: int | None = None,
    webui_data_dir: Path | None = None,
    webui_image: str | None = None,
    cwd: Path | None = None,
) -> Config:
    """Build a :class:`Config` by merging all sources in priority order.

    Args:
        backend: CLI-provided backend override.
        models_dir: CLI-provided models directory override.
        cache_dir: CLI-provided cache directory override.
        hf_token: CLI-provided HuggingFace token override.
        docker_image: CLI-provided Docker image override.
        server_host: CLI-provided server host override.
        server_port: CLI-provided server port override.
        webui_host: CLI-provided WebUI host override.
        webui_port: CLI-provided WebUI port override.
        webui_data_dir: CLI-provided WebUI data directory override.
        webui_image: CLI-provided WebUI Docker image override.
        cwd: Directory to search for ``ov-manager.toml``. Defaults to cwd.

    Returns:
        Fully resolved :class:`Config` instance.
    """
    toml = _load_toml(cwd)

    def _resolve_str(cli_val: str | None, env_key: str, toml_key: str, default: str) -> str:
        if cli_val is not None:
            return cli_val
        env_val = os.environ.get(env_key)
        if env_val:
            return env_val
        toml_val = toml.get(toml_key)
        if toml_val:
            return str(toml_val)
        return default

    def _resolve_int(cli_val: int | None, env_key: str, toml_key: str, default: int) -> int:
        if cli_val is not None:
            return cli_val
        env_val = os.environ.get(env_key)
        if env_val:
            return int(env_val)
        toml_val = toml.get(toml_key)
        if toml_val is not None:
            return int(toml_val)
        return default

    def _resolve_path(cli_val: Path | None, env_key: str, toml_key: str, default: Path) -> Path:
        if cli_val is not None:
            return cli_val.expanduser().resolve()
        env_val = os.environ.get(env_key)
        if env_val:
            return Path(env_val).expanduser().resolve()
        toml_val = toml.get(toml_key)
        if toml_val:
            return Path(str(toml_val)).expanduser().resolve()
        return default.expanduser().resolve()

    def _resolve_optional_path(
        cli_val: Path | None,
        env_key: str,
        toml_key: str,
    ) -> Path | None:
        if cli_val is not None:
            return cli_val.expanduser().resolve()
        env_val = os.environ.get(env_key)
        if env_val:
            return Path(env_val).expanduser().resolve()
        toml_val = toml.get(toml_key)
        if toml_val:
            return Path(str(toml_val)).expanduser().resolve()
        return None

    def _resolve_token() -> str | None:
        if hf_token is not None:
            return hf_token
        for key in ("OVMGR_HF_TOKEN", "HF_TOKEN"):
            val = os.environ.get(key)
            if val:
                return val
        toml_val = toml.get("hf_token")
        if toml_val:
            return str(toml_val)
        return None

    resolved_backend = _resolve_str(backend, "OVMGR_BACKEND", "backend", DEFAULT_BACKEND)
    if resolved_backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend {resolved_backend!r}. Must be one of: {', '.join(VALID_BACKENDS)}"
        )

    # Resolve "auto" to a concrete backend: baremetal > docker
    if resolved_backend == "auto":
        if shutil.which("ovms"):
            resolved_backend = "baremetal"
        elif shutil.which("docker"):
            resolved_backend = "docker"
        else:
            raise click.ClickException(
                "No OVMS backend found. Install 'ovms' on $PATH for bare-metal, "
                "or install Docker for the docker backend. "
                "You can also set --backend explicitly."
            )

    return Config(
        backend=resolved_backend,
        models_dir=_resolve_path(models_dir, "OVMGR_MODELS_DIR", "models_dir", DEFAULT_MODELS_DIR),
        cache_dir=_resolve_path(cache_dir, "OVMGR_CACHE_DIR", "cache_dir", DEFAULT_CACHE_DIR),
        hf_token=_resolve_token(),
        docker_image=_resolve_str(
            docker_image, "OVMGR_DOCKER_IMAGE", "docker_image", DEFAULT_DOCKER_IMAGE
        ),
        server_host=_resolve_str(
            server_host, "OVMGR_SERVER_HOST", "server_host", DEFAULT_SERVER_HOST
        ),
        server_port=_resolve_int(
            server_port, "OVMGR_SERVER_PORT", "server_port", DEFAULT_SERVER_PORT
        ),
        webui_host=_resolve_str(webui_host, "OVMGR_WEBUI_HOST", "webui_host", DEFAULT_WEBUI_HOST),
        webui_port=_resolve_int(webui_port, "OVMGR_WEBUI_PORT", "webui_port", DEFAULT_WEBUI_PORT),
        webui_data_dir=_resolve_optional_path(
            webui_data_dir, "OVMGR_WEBUI_DATA_DIR", "webui_data_dir"
        ),
        webui_image=_resolve_str(
            webui_image, "OVMGR_WEBUI_IMAGE", "webui_image", DEFAULT_WEBUI_IMAGE
        ),
    )
