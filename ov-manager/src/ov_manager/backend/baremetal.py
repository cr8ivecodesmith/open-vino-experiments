"""Bare-metal backend: invokes the local ``ovms`` binary directly."""

from __future__ import annotations

import shutil
import subprocess

import click

from ov_manager.backend.base import PullParams, RegisterParams, RemoveParams, ServeParams


class BaremetalBackend:
    """Runs OVMS commands using the ``ovms`` binary found on ``$PATH``."""

    BINARY = "ovms"

    def _ovms(self, args: list[str], env: dict[str, str] | None = None) -> None:
        """Run the ``ovms`` binary with *args*, streaming output to the terminal.

        Args:
            args: Command-line arguments to pass after ``ovms``.
            env: Optional environment variable overrides.

        Raises:
            click.ClickException: If the binary is not found or exits non-zero.
        """
        binary = shutil.which(self.BINARY)
        if binary is None:
            raise click.ClickException(
                f"'{self.BINARY}' not found on $PATH. Install OVMS or use --backend docker."
            )

        cmd = [binary, *args]
        click.echo(f"$ {' '.join(cmd)}")

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise click.ClickException(f"'{self.BINARY}' exited with code {result.returncode}.")

    def pull(self, params: PullParams) -> None:
        """Download a model via ``ovms --pull``.

        Args:
            params: Pull parameters.
        """
        import os

        env = os.environ.copy()
        if params.hf_token:
            env["HF_TOKEN"] = params.hf_token

        args = [
            "--pull",
            "--source_model",
            params.source_model,
            "--model_repository_path",
            str(params.model_repository_path),
            "--model_name",
            params.model_name,
            "--target_device",
            params.target_device,
            "--task",
            params.task,
        ]

        if params.weight_format is not None:
            args += ["--weight-format", params.weight_format]

        if params.overwrite:
            args.append("--overwrite_models")
        if params.pipeline_type:
            args += ["--pipeline_type", params.pipeline_type]
        if params.cache_size is not None:
            args += ["--cache_size", str(params.cache_size)]
        if params.extra_quantization_params:
            args += ["--extra_quantization_params", params.extra_quantization_params]

        self._ovms(args, env=env)

    def register(self, params: RegisterParams) -> None:
        """Register a model via ``ovms --add_to_config``.

        Args:
            params: Registration parameters.
        """
        args = [
            "--add_to_config",
            "--config_path",
            str(params.config_json_path),
            "--model_repository_path",
            str(params.model_repository_path),
            "--model_name",
            params.model_name,
            "--model_path",
            params.model_path,
        ]
        self._ovms(args)

    def serve(self, params: ServeParams) -> None:
        """Start the OVMS server via bare-metal ``ovms`` binary.

        Both foreground and background modes write a PID file to the XDG
        runtime directory so that ``server status`` and ``server stop`` can
        detect the running process.

        In foreground mode, blocks on ``proc.wait()`` and cleans up the PID
        file when the process exits. In background mode, the PID file persists
        until ``server stop`` removes it.

        Args:
            params: Serve parameters.
        """
        import subprocess as _subprocess

        from ov_manager.runtime import get_runtime_dir, remove_pidfile, write_pidfile

        binary = shutil.which(self.BINARY)
        if binary is None:
            raise click.ClickException(
                f"'{self.BINARY}' not found on $PATH. Install OVMS or use --backend docker."
            )

        args = [
            binary,
            "--rest_port",
            str(params.port),
            "--config_path",
            str(params.config_json_path),
        ]

        click.echo(f"$ {' '.join(args)}")

        proc = _subprocess.Popen(args)
        pidfile = get_runtime_dir() / "ovmgr-server.pid"
        write_pidfile(pidfile, proc.pid)

        if params.background:
            click.echo(f"OVMS started in background (PID {proc.pid}).")
        else:
            try:
                proc.wait()
            finally:
                remove_pidfile(pidfile)
            if proc.returncode != 0:
                raise click.ClickException(f"'{self.BINARY}' exited with code {proc.returncode}.")

    def remove(self, params: RemoveParams) -> None:
        """Unregister a model via ``ovms --remove_from_config``.

        Args:
            params: Remove parameters.
        """
        args = [
            "--remove_from_config",
            "--config_path",
            str(params.config_json_path),
            "--model_name",
            params.model_name,
        ]
        self._ovms(args)
