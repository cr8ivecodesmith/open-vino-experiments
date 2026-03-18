"""``ov-manage server`` command group: run, stop, status.

Commands::

    ov-manage server run [HOST:PORT] [-d] [--webui [HOST:PORT]] ...
    ov-manage server stop [--service ovms|webui]
    ov-manage server status
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ov_manager.backend import get_backend
from ov_manager.backend.base import ServeParams
from ov_manager.config import Config
from ov_manager.runtime import get_runtime_dir, read_pidfile, remove_pidfile

console = Console()

OVMS_CONTAINER = "ovmgr-server"
WEBUI_CONTAINER = "ovmgr-webui"


# ---------------------------------------------------------------------------
# Helpers — WebUI
# ---------------------------------------------------------------------------


def start_webui(
    *,
    host: str,
    port: int,
    ovms_port: int,
    image: str,
    data_dir: Path | None,
    background: bool,
) -> None:
    """Start the Open WebUI Docker container.

    Args:
        host: Host address to bind the WebUI port to.
        port: Host port for the WebUI.
        ovms_port: OVMS REST API port (used to build the OpenAI base URL).
        image: Docker image for Open WebUI.
        data_dir: Optional host path to bind-mount for persistent data.
        background: If True, run with ``-d`` (detached).
    """
    if shutil.which("docker") is None:
        raise click.ClickException(
            "'docker' not found on $PATH. Docker is required for Open WebUI."
        )

    cmd = [
        "docker",
        "run",
        "--rm",
        "--name",
        WEBUI_CONTAINER,
        "--add-host=host.docker.internal:host-gateway",
        "-p",
        f"{host}:{port}:8080",
        "-e",
        "ENABLE_OLLAMA_API=False",
        "-e",
        f"OPENAI_API_BASE_URL=http://host.docker.internal:{ovms_port}/v3",
        "-e",
        "OPENAI_API_KEY=unused",
    ]

    if background:
        cmd.insert(2, "-d")

    if data_dir is not None:
        cmd += ["-v", f"{data_dir}:/app/backend/data"]

    cmd.append(image)

    click.echo(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise click.ClickException(f"Open WebUI container exited with code {result.returncode}.")


# ---------------------------------------------------------------------------
# Helpers — Stop
# ---------------------------------------------------------------------------


def _stop_docker_container(name: str) -> bool:
    """Stop a Docker container by name. Returns True if it was stopped."""
    result = subprocess.run(
        ["docker", "stop", name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _stop_baremetal_ovms() -> bool:
    """Stop a bare-metal OVMS process via its pidfile. Returns True if killed."""
    runtime = get_runtime_dir()
    pidfile = runtime / "ovmgr-server.pid"
    pid = read_pidfile(pidfile)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    remove_pidfile(pidfile)
    return True


def stop_services(service: str | None, backend: str) -> None:
    """Stop running services.

    Args:
        service: ``"ovms"``, ``"webui"``, or ``None`` for both.
        backend: The backend type (``"baremetal"`` or ``"docker"``).
    """
    stopped: list[str] = []

    if service in (None, "ovms"):
        if backend == "baremetal":
            if _stop_baremetal_ovms():
                stopped.append("ovmgr-server (baremetal)")
            else:
                click.echo("OVMS: no pidfile found or process not running.")
        else:
            if _stop_docker_container(OVMS_CONTAINER):
                stopped.append(OVMS_CONTAINER)
            else:
                click.echo(f"{OVMS_CONTAINER}: not running or not found.")

    if service in (None, "webui"):
        if _stop_docker_container(WEBUI_CONTAINER):
            stopped.append(WEBUI_CONTAINER)
        else:
            click.echo(f"{WEBUI_CONTAINER}: not running or not found.")

    if stopped:
        console.print(
            Panel(
                "[bold green]Stopped:[/bold green] " + ", ".join(stopped),
                title="Done",
                border_style="green",
            )
        )
    else:
        click.echo("No services were running.")


# ---------------------------------------------------------------------------
# Helpers — Status
# ---------------------------------------------------------------------------


def _docker_container_running(name: str) -> bool:
    """Check if a Docker container is running."""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _baremetal_pid_alive(pidfile: Path) -> int | None:
    """Return PID if the process is alive, else None."""
    pid = read_pidfile(pidfile)
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
        return pid
    except (ProcessLookupError, PermissionError):
        return None


def get_service_statuses(backend: str) -> list[dict[str, Any]]:
    """Get the status of all managed services.

    Args:
        backend: The backend type.

    Returns:
        List of dicts with keys: name, status, backend_type.
    """
    statuses: list[dict[str, Any]] = []

    # OVMS
    if backend == "baremetal":
        runtime = get_runtime_dir()
        pidfile = runtime / "ovmgr-server.pid"
        pid = _baremetal_pid_alive(pidfile)
        statuses.append(
            {
                "name": OVMS_CONTAINER,
                "status": "running" if pid else "stopped",
                "backend_type": "baremetal",
                "detail": f"PID {pid}" if pid else "",
            }
        )
    else:
        running = _docker_container_running(OVMS_CONTAINER)
        statuses.append(
            {
                "name": OVMS_CONTAINER,
                "status": "running" if running else "stopped",
                "backend_type": "docker",
                "detail": "",
            }
        )

    # WebUI — always docker
    running = _docker_container_running(WEBUI_CONTAINER)
    statuses.append(
        {
            "name": WEBUI_CONTAINER,
            "status": "running" if running else "stopped",
            "backend_type": "docker",
            "detail": "",
        }
    )

    return statuses


# ---------------------------------------------------------------------------
# Helpers — Address parsing
# ---------------------------------------------------------------------------


def _parse_address(raw: str, default_host: str, default_port: int) -> tuple[str, int]:
    """Parse an optional ``HOST:PORT`` string, filling in defaults."""
    parts = raw.rsplit(":", 1)
    if len(parts) == 2:
        return parts[0] or default_host, int(parts[1])
    return default_host, int(parts[0])


# ---------------------------------------------------------------------------
# Helpers — run_server sub-flows
# ---------------------------------------------------------------------------


def _run_background(
    *,
    config: Config,
    serve_params: ServeParams,
    webui_address: str | None,
    webui_host: str,
    webui_port: int,
    resolved_webui_data: Path | None,
    resolved_webui_image: str,
) -> None:
    """Start OVMS (and optionally WebUI) in the background and print a summary."""
    console.rule("[bold]Starting OVMS (background)[/bold]")
    get_backend(config).serve(serve_params)

    if webui_address is not None:
        console.rule("[bold]Starting Open WebUI (background)[/bold]")
        start_webui(
            host=webui_host,
            port=webui_port,
            ovms_port=serve_params.port,
            image=resolved_webui_image,
            data_dir=resolved_webui_data,
            background=True,
        )

    webui_line = f"WebUI: [cyan]{webui_host}:{webui_port}[/cyan]\n" if webui_address else ""
    console.print(
        Panel(
            f"[bold green]Services started in background.[/bold green]\n"
            f"OVMS: [cyan]{serve_params.host}:{serve_params.port}[/cyan]\n"
            + webui_line
            + "\nRun [bold]ov-manage server stop[/bold] to stop.",
            title="Running",
            border_style="green",
        )
    )


def _run_foreground(
    *,
    config: Config,
    serve_params: ServeParams,
    webui_address: str | None,
    webui_host: str,
    webui_port: int,
    resolved_webui_data: Path | None,
    resolved_webui_image: str,
) -> None:
    """Start OVMS in the foreground with signal handling for clean shutdown."""
    webui_started = False

    def _shutdown(sig: int, frame: Any) -> None:
        console.print("\n[yellow]Shutting down...[/yellow]")
        stop_services(service=None, backend=config.backend)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if webui_address is not None:
        # Start WebUI in background (Docker -d) so we can block on OVMS
        console.rule("[bold]Starting Open WebUI (background)[/bold]")
        start_webui(
            host=webui_host,
            port=webui_port,
            ovms_port=serve_params.port,
            image=resolved_webui_image,
            data_dir=resolved_webui_data,
            background=True,
        )
        webui_started = True

    console.rule(f"[bold]Starting OVMS on {serve_params.host}:{serve_params.port}[/bold]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]")

    try:
        get_backend(config).serve(serve_params)
    finally:
        if webui_started:
            _stop_docker_container(WEBUI_CONTAINER)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def server() -> None:
    """Manage the OVMS and Open WebUI servers."""


@server.command("run")
@click.argument("address", default=None, required=False)
@click.option("-d", "--detach", is_flag=True, default=False, help="Run in the background.")
@click.option(
    "--webui",
    "webui_address",
    default=None,
    help=(
        "Start Open WebUI. Pass 'default' for default address or HOST:PORT to override. "
        "Omit to skip WebUI."
    ),
)
@click.option("--webui-data", type=click.Path(path_type=Path), default=None, help="WebUI data dir.")
@click.option("--webui-image", default=None, help="WebUI Docker image.")
@click.pass_obj
def run_server(
    config: Config,
    address: str | None,
    detach: bool,
    webui_address: str | None,
    webui_data: Path | None,
    webui_image: str | None,
) -> None:
    r"""Start the OVMS server (and optionally Open WebUI).

    \b
    ADDRESS is optional HOST:PORT for OVMS (default: 127.0.0.1:8100).
    """
    ovms_host, ovms_port = (
        _parse_address(address, config.server_host, config.server_port)
        if address
        else (config.server_host, config.server_port)
    )
    webui_host, webui_port = (
        _parse_address(webui_address, config.webui_host, config.webui_port)
        if webui_address and webui_address != "default"
        else (config.webui_host, config.webui_port)
    )

    resolved_webui_data = webui_data or config.webui_data_dir
    resolved_webui_image = webui_image or config.webui_image

    if not config.config_json_path.exists():
        raise click.ClickException(
            f"No config.json found at {config.config_json_path}. "
            "Run 'ov-manage models get' first to download and register a model."
        )

    serve_params = ServeParams(
        config_json_path=config.config_json_path,
        models_dir=config.models_dir,
        host=ovms_host,
        port=ovms_port,
        background=detach,
    )

    kwargs: dict[str, Any] = dict(
        config=config,
        serve_params=serve_params,
        webui_address=webui_address,
        webui_host=webui_host,
        webui_port=webui_port,
        resolved_webui_data=resolved_webui_data,
        resolved_webui_image=resolved_webui_image,
    )
    if detach:
        _run_background(**kwargs)
    else:
        _run_foreground(**kwargs)


@server.command("stop")
@click.option(
    "--service",
    type=click.Choice(["ovms", "webui"]),
    default=None,
    help="Stop a specific service. Default: stop all.",
)
@click.pass_obj
def stop_server(config: Config, service: str | None) -> None:
    """Stop running OVMS and/or Open WebUI services."""
    stop_services(service=service, backend=config.backend)


@server.command("status")
@click.pass_obj
def server_status(config: Config) -> None:
    """Show status of OVMS and Open WebUI services."""
    statuses = get_service_statuses(backend=config.backend)

    table = Table(title="Service Status", show_lines=True)
    table.add_column("Service", style="bold cyan")
    table.add_column("Status")
    table.add_column("Backend", style="dim")
    table.add_column("Detail", style="dim")

    for s in statuses:
        status_cell = "[green]running[/green]" if s["status"] == "running" else "[red]stopped[/red]"
        table.add_row(s["name"], status_cell, s["backend_type"], s["detail"])

    console.print(table)
