"""Backend resolution for ov-manager.

Exports :func:`get_backend` which returns the appropriate backend instance
based on the resolved :class:`~ov_manager.config.Config`.
"""

from __future__ import annotations

import shutil

import click

from ov_manager.backend.baremetal import BaremetalBackend
from ov_manager.backend.base import Backend
from ov_manager.backend.docker import DockerBackend
from ov_manager.config import Config


def get_backend(config: Config) -> Backend:
    """Return the backend instance for *config*.

    For ``auto``, prefers bare-metal if ``ovms`` is on ``$PATH``, then falls
    back to Docker.

    Args:
        config: Resolved configuration.

    Returns:
        A :class:`~ov_manager.backend.base.Backend` implementation.

    Raises:
        click.ClickException: If ``auto`` detection finds neither ``ovms``
            nor ``docker``.
    """
    backend = config.backend

    if backend == "baremetal":
        return BaremetalBackend()

    if backend == "docker":
        return DockerBackend(image=config.docker_image)

    # auto
    if shutil.which("ovms"):
        return BaremetalBackend()
    if shutil.which("docker"):
        return DockerBackend(image=config.docker_image)

    raise click.ClickException(
        "No OVMS backend found. Install 'ovms' on $PATH for bare-metal, "
        "or install Docker for the docker backend. "
        "You can also set --backend explicitly."
    )
