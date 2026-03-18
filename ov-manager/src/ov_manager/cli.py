"""CLI entry point for ov-manage.

Usage::

    ov-manage models list
    ov-manage models list --json
    ov-manage --backend docker models get OpenVINO/whisper-base-fp16-ov --task text_generation
    ov-manage --models-dir /data/models models get OpenVINO/FLUX.1-schnell-int4-ov --task image_generation
"""

from __future__ import annotations

from pathlib import Path

import click

from ov_manager.commands.models import models
from ov_manager.config import VALID_BACKENDS, resolve


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--backend",
    type=click.Choice(list(VALID_BACKENDS)),
    default=None,
    help=(
        "Execution backend. "
        "'auto' uses bare-metal ovms if on $PATH, else docker. "
        "[env: OVMGR_BACKEND]"
    ),
)
@click.option(
    "--models-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Model repository directory. [env: OVMGR_MODELS_DIR]",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="OVMS compiled model cache directory. [env: OVMGR_CACHE_DIR]",
)
@click.option(
    "--docker-image",
    default=None,
    help="Docker image to use with the docker backend. [env: OVMGR_DOCKER_IMAGE]",
)
@click.pass_context
def main(
    ctx: click.Context,
    backend: str | None,
    models_dir: Path | None,
    cache_dir: Path | None,
    docker_image: str | None,
) -> None:
    """OpenVINO Model Server manager.

    Manage models for use with OpenVINO Model Server (OVMS): download from
    HuggingFace, register in the model repository, and inspect what's available.

    Configuration is resolved in priority order:
    CLI flags > OVMGR_* env vars > ov-manager.toml > defaults.

    The config file is located by searching upward from the current directory
    to the git root, then falling back to ~/.config/ov-manager/ov-manager.toml.
    Override with OVMGR_CONFIG=/path/to/ov-manager.toml.
    """
    ctx.ensure_object(dict)
    ctx.obj = resolve(
        backend=backend,
        models_dir=models_dir,
        cache_dir=cache_dir,
        docker_image=docker_image,
    )


main.add_command(models)
