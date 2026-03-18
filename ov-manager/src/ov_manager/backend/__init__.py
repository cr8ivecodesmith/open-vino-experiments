"""Backend resolution for ov-manager.

Exports :func:`get_backend` which returns the appropriate backend instance
based on the resolved :class:`~ov_manager.config.Config`.

Note: ``config.backend`` is always ``"baremetal"`` or ``"docker"`` by the
time it reaches this module — ``"auto"`` is resolved in
:func:`~ov_manager.config.resolve`.
"""

from __future__ import annotations

import click

from ov_manager.backend.baremetal import BaremetalBackend
from ov_manager.backend.base import Backend
from ov_manager.backend.docker import DockerBackend
from ov_manager.config import Config


def get_backend(config: Config) -> Backend:
    """Return the backend instance for *config*.

    Args:
        config: Resolved configuration. ``config.backend`` must be
            ``"baremetal"`` or ``"docker"`` (``"auto"`` is resolved earlier
            in :func:`~ov_manager.config.resolve`).

    Returns:
        A :class:`~ov_manager.backend.base.Backend` implementation.

    Raises:
        click.ClickException: If the backend value is unrecognised.
    """
    if config.backend == "baremetal":
        return BaremetalBackend()

    if config.backend == "docker":
        return DockerBackend(image=config.docker_image)

    raise click.ClickException(
        f"Unknown backend {config.backend!r}. Expected 'baremetal' or 'docker'."
    )
