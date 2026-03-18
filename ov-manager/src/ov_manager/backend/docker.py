"""Docker backend: invokes ``ovms`` via ``docker run openvino/model_server``."""

from __future__ import annotations

import os
import shutil
import subprocess

import click

from ov_manager.backend.base import PullParams, RegisterParams, RemoveParams, ServeParams


class DockerBackend:
    """Runs OVMS commands by wrapping ``docker run openvino/model_server``.

    The container is always run with:
    - ``--rm`` — ephemeral, cleaned up after each command
    - ``-u $(id -u):$(id -g)`` — avoid root-owned files in the model dir
    - ``-v <models_dir>:/models`` — bind-mount the model repository
    - ``-e HF_TOKEN`` — only when a token is available
    """

    def __init__(self, image: str = "openvino/model_server:latest") -> None:
        """Initialise the Docker backend.

        Args:
            image: Docker image to use for OVMS invocations.
        """
        self.image = image

    def _docker(self, ovms_args: list[str], models_dir: str, hf_token: str | None = None) -> None:
        """Run ``docker run ... openvino/model_server <ovms_args>``.

        Args:
            ovms_args: Arguments forwarded directly to the OVMS binary inside
                the container.
            models_dir: Host path to bind-mount as ``/models`` inside the container.
            hf_token: Optional HuggingFace token to inject as ``HF_TOKEN``.

        Raises:
            click.ClickException: If ``docker`` is not found or the container
                exits non-zero.
        """
        if shutil.which("docker") is None:
            raise click.ClickException(
                "'docker' not found on $PATH. Install Docker or use --backend baremetal."
            )

        uid = os.getuid()
        gid = os.getgid()

        cmd = [
            "docker",
            "run",
            "--rm",
            "-u",
            f"{uid}:{gid}",
            "-v",
            f"{models_dir}:/models",
        ]

        if hf_token:
            cmd += ["-e", f"HF_TOKEN={hf_token}"]

        cmd += [self.image, *ovms_args]

        click.echo(f"$ {' '.join(cmd)}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise click.ClickException(f"Docker container exited with code {result.returncode}.")

    def pull(self, params: PullParams) -> None:
        """Download a model via ``docker run ... --pull``.

        The model repository path is always ``/models`` inside the container,
        since the host ``models_dir`` is bind-mounted there.

        Args:
            params: Pull parameters.
        """
        ovms_args = [
            "--pull",
            "--source_model",
            params.source_model,
            "--model_repository_path",
            "/models",
            "--model_name",
            params.model_name,
            "--target_device",
            params.target_device,
            "--task",
            params.task,
        ]

        if params.weight_format is not None:
            ovms_args += ["--weight-format", params.weight_format]

        if params.overwrite:
            ovms_args.append("--overwrite_models")
        if params.pipeline_type:
            ovms_args += ["--pipeline_type", params.pipeline_type]
        if params.cache_size is not None:
            ovms_args += ["--cache_size", str(params.cache_size)]
        if params.extra_quantization_params:
            ovms_args += ["--extra_quantization_params", params.extra_quantization_params]

        self._docker(ovms_args, str(params.model_repository_path), hf_token=params.hf_token)

    def register(self, params: RegisterParams) -> None:
        """Register a model via ``docker run ... --add_to_config``.

        Args:
            params: Registration parameters.
        """
        ovms_args = [
            "--add_to_config",
            "--config_path",
            "/models/config.json",
            "--model_repository_path",
            "/models",
            "--model_name",
            params.model_name,
            "--model_path",
            params.model_path,
        ]
        self._docker(ovms_args, str(params.model_repository_path))

    def remove(self, params: RemoveParams) -> None:
        """Unregister a model via ``docker run ... --remove_from_config``.

        Args:
            params: Remove parameters.
        """
        ovms_args = [
            "--remove_from_config",
            "--config_path",
            "/models/config.json",
            "--model_name",
            params.model_name,
        ]
        self._docker(ovms_args, str(params.config_json_path.parent))

    def serve(self, params: ServeParams) -> None:
        """Start the OVMS server via Docker container.

        Uses container name ``ovmgr-server`` for lifecycle management.

        Args:
            params: Serve parameters.
        """
        if shutil.which("docker") is None:
            raise click.ClickException(
                "'docker' not found on $PATH. Install Docker or use --backend baremetal."
            )

        uid = os.getuid()
        gid = os.getgid()

        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            "ovmgr-server",
            "-u",
            f"{uid}:{gid}",
            "-v",
            f"{params.models_dir}:/models",
            "-p",
            f"{params.host}:{params.port}:8000",
        ]

        if params.background:
            cmd.insert(2, "-d")

        cmd += [
            self.image,
            "--rest_port",
            "8000",
            "--config_path",
            "/models/config.json",
        ]

        click.echo(f"$ {' '.join(cmd)}")

        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise click.ClickException(f"Docker container exited with code {result.returncode}.")
