"""Tests for server run/stop/status — backend serve() and commands.

London style: mock subprocess/docker calls. Chicago for pure logic.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

base_mod = importlib.import_module("ov_manager.backend.base")
baremetal_mod = importlib.import_module("ov_manager.backend.baremetal")
docker_mod = importlib.import_module("ov_manager.backend.docker")
server_mod = importlib.import_module("ov_manager.commands.server")
runtime_mod = importlib.import_module("ov_manager.runtime")
config_mod = importlib.import_module("ov_manager.config")


# ---------------------------------------------------------------------------
# ServeParams dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_serve_params_exists():
    """ServeParams dataclass exists on base module."""
    assert hasattr(base_mod, "ServeParams")
    params = base_mod.ServeParams(
        config_json_path=Path("/models/config.json"),
        models_dir=Path("/models"),
        host="127.0.0.1",
        port=8100,
        background=False,
    )
    assert params.host == "127.0.0.1"
    assert params.port == 8100
    assert params.background is False


# ---------------------------------------------------------------------------
# Baremetal serve()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_serve_foreground_baremetal_calls_ovms(mocker, tmp_path):
    """Bare-metal foreground serve() spawns Popen with correct args and waits."""
    mocker.patch("shutil.which", return_value="/usr/bin/ovms")
    mock_proc = MagicMock()
    mock_proc.pid = 99
    mock_proc.wait.return_value = 0
    mock_proc.returncode = 0
    mocker.patch("subprocess.Popen", return_value=mock_proc)
    mocker.patch("ov_manager.runtime.get_runtime_dir", return_value=tmp_path)

    backend = baremetal_mod.BaremetalBackend()
    params = base_mod.ServeParams(
        config_json_path=Path("/models/config.json"),
        models_dir=Path("/models"),
        host="127.0.0.1",
        port=8100,
        background=False,
    )

    backend.serve(params)

    # Popen called with correct args

    subprocess.Popen.assert_called_once()
    cmd = subprocess.Popen.call_args[0][0]
    assert "--rest_port" in cmd
    assert "8100" in cmd
    assert "--config_path" in cmd
    assert "/models/config.json" in cmd

    # Process was waited on
    mock_proc.wait.assert_called_once()


@pytest.mark.unit
def test_serve_foreground_baremetal_writes_and_cleans_pidfile(mocker, tmp_path):
    """Bare-metal foreground serve() writes pidfile during run and cleans up after."""
    mocker.patch("shutil.which", return_value="/usr/bin/ovms")
    mock_proc = MagicMock()
    mock_proc.pid = 55
    mock_proc.returncode = 0

    pidfile = tmp_path / "ovmgr-server.pid"

    def fake_wait():
        # While OVMS is "running", pidfile should exist
        assert pidfile.exists(), "pidfile should exist while process is running"
        assert pidfile.read_text().strip() == "55"
        return 0

    mock_proc.wait.side_effect = fake_wait
    mocker.patch("subprocess.Popen", return_value=mock_proc)
    mocker.patch("ov_manager.runtime.get_runtime_dir", return_value=tmp_path)

    backend = baremetal_mod.BaremetalBackend()
    params = base_mod.ServeParams(
        config_json_path=Path("/models/config.json"),
        models_dir=Path("/models"),
        host="127.0.0.1",
        port=8100,
        background=False,
    )

    backend.serve(params)

    # After serve() returns, pidfile should be cleaned up
    assert not pidfile.exists(), "pidfile should be removed after process exits"


@pytest.mark.unit
def test_serve_background_baremetal_writes_pidfile(mocker, tmp_path):
    """Bare-metal background serve() spawns Popen and writes pidfile."""
    mocker.patch("shutil.which", return_value="/usr/bin/ovms")
    mock_popen = MagicMock()
    mock_popen.pid = 42
    mocker.patch("subprocess.Popen", return_value=mock_popen)
    mocker.patch("ov_manager.runtime.get_runtime_dir", return_value=tmp_path)

    backend = baremetal_mod.BaremetalBackend()
    params = base_mod.ServeParams(
        config_json_path=Path("/models/config.json"),
        models_dir=Path("/models"),
        host="127.0.0.1",
        port=8100,
        background=True,
    )

    backend.serve(params)

    # Check pidfile was written
    pidfile = tmp_path / "ovmgr-server.pid"
    assert pidfile.exists()
    assert pidfile.read_text().strip() == "42"


# ---------------------------------------------------------------------------
# Docker serve()
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_serve_foreground_docker_calls_docker_run(mocker):
    """Docker foreground serve() runs docker with correct args."""
    mocker.patch("shutil.which", return_value="/usr/bin/docker")
    mocker.patch("os.getuid", return_value=1000)
    mocker.patch("os.getgid", return_value=1000)
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    backend = docker_mod.DockerBackend(image="openvino/model_server:latest")
    params = base_mod.ServeParams(
        config_json_path=Path("/models/config.json"),
        models_dir=Path("/models"),
        host="127.0.0.1",
        port=8100,
        background=False,
    )

    backend.serve(params)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "docker" in cmd[0]
    assert "--name" in cmd
    assert "ovmgr-server" in cmd
    assert "-p" in cmd
    # Port mapping: host_port:container_port
    port_arg_idx = cmd.index("-p") + 1
    assert "8100:8000" in cmd[port_arg_idx]


@pytest.mark.unit
def test_serve_background_docker_adds_d_flag(mocker):
    """Docker background serve() adds -d flag to docker run."""
    mocker.patch("shutil.which", return_value="/usr/bin/docker")
    mocker.patch("os.getuid", return_value=1000)
    mocker.patch("os.getgid", return_value=1000)
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    backend = docker_mod.DockerBackend(image="openvino/model_server:latest")
    params = base_mod.ServeParams(
        config_json_path=Path("/models/config.json"),
        models_dir=Path("/models"),
        host="127.0.0.1",
        port=8100,
        background=True,
    )

    backend.serve(params)

    cmd = mock_run.call_args[0][0]
    assert "-d" in cmd


# ---------------------------------------------------------------------------
# WebUI Docker launch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_start_webui_env_vars_correct(mocker):
    """WebUI container gets correct env vars for OVMS connection."""
    mocker.patch("shutil.which", return_value="/usr/bin/docker")
    mocker.patch("os.getuid", return_value=1000)
    mocker.patch("os.getgid", return_value=1000)
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    server_mod.start_webui(
        host="127.0.0.1",
        port=3100,
        ovms_port=8100,
        image="ghcr.io/open-webui/open-webui:main",
        data_dir=None,
        background=False,
    )

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    cmd_str = " ".join(cmd)

    assert "ENABLE_OLLAMA_API=False" in cmd_str
    assert "OPENAI_API_BASE_URL=http://host.docker.internal:8100/v3" in cmd_str
    assert "OPENAI_API_KEY=unused" in cmd_str


@pytest.mark.unit
def test_start_webui_add_host_on_linux(mocker):
    """WebUI docker run includes --add-host for Linux connectivity."""
    mocker.patch("shutil.which", return_value="/usr/bin/docker")
    mocker.patch("os.getuid", return_value=1000)
    mocker.patch("os.getgid", return_value=1000)
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    server_mod.start_webui(
        host="127.0.0.1",
        port=3100,
        ovms_port=8100,
        image="ghcr.io/open-webui/open-webui:main",
        data_dir=None,
        background=False,
    )

    cmd = mock_run.call_args[0][0]
    assert "--add-host=host.docker.internal:host-gateway" in cmd


@pytest.mark.unit
def test_start_webui_data_volume_mounted(mocker):
    """WebUI data dir is bind-mounted when provided."""
    mocker.patch("shutil.which", return_value="/usr/bin/docker")
    mocker.patch("os.getuid", return_value=1000)
    mocker.patch("os.getgid", return_value=1000)
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    server_mod.start_webui(
        host="127.0.0.1",
        port=3100,
        ovms_port=8100,
        image="ghcr.io/open-webui/open-webui:main",
        data_dir=Path("/srv/webui-data"),
        background=False,
    )

    cmd = mock_run.call_args[0][0]
    cmd_str = " ".join(cmd)
    assert "/srv/webui-data:/app/backend/data" in cmd_str


@pytest.mark.unit
def test_start_webui_no_data_volume_when_none(mocker):
    """No data volume is mounted when data_dir is None."""
    mocker.patch("shutil.which", return_value="/usr/bin/docker")
    mocker.patch("os.getuid", return_value=1000)
    mocker.patch("os.getgid", return_value=1000)
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    server_mod.start_webui(
        host="127.0.0.1",
        port=3100,
        ovms_port=8100,
        image="ghcr.io/open-webui/open-webui:main",
        data_dir=None,
        background=False,
    )

    cmd = mock_run.call_args[0][0]
    # The only -v should be absent (no data volume)
    v_indices = [i for i, arg in enumerate(cmd) if arg == "-v"]
    for idx in v_indices:
        assert "/app/backend/data" not in cmd[idx + 1]


@pytest.mark.unit
def test_start_webui_custom_image(mocker):
    """Custom WebUI image is used when provided."""
    mocker.patch("shutil.which", return_value="/usr/bin/docker")
    mocker.patch("os.getuid", return_value=1000)
    mocker.patch("os.getgid", return_value=1000)
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    server_mod.start_webui(
        host="127.0.0.1",
        port=3100,
        ovms_port=8100,
        image="ghcr.io/open-webui/open-webui:cuda",
        data_dir=None,
        background=False,
    )

    cmd = mock_run.call_args[0][0]
    assert "ghcr.io/open-webui/open-webui:cuda" in cmd


# ---------------------------------------------------------------------------
# Stop
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stop_all_stops_both(mocker):
    """Server stop (no --service) stops both containers."""
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))
    mocker.patch("ov_manager.runtime.get_runtime_dir", return_value=Path("/tmp/rt"))
    mocker.patch("ov_manager.runtime.read_pidfile", return_value=None)

    server_mod.stop_services(service=None, backend="docker")

    # Should have tried to stop both containers
    calls = mock_run.call_args_list
    stop_cmds = [c for c in calls if "stop" in c[0][0]]
    container_names = []
    for c in stop_cmds:
        container_names.extend(c[0][0])
    assert "ovmgr-server" in container_names
    assert "ovmgr-webui" in container_names


@pytest.mark.unit
def test_stop_selective_ovms_only(mocker):
    """Server stop --service ovms only stops ovmgr-server."""
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))
    mocker.patch("ov_manager.runtime.get_runtime_dir", return_value=Path("/tmp/rt"))
    mocker.patch("ov_manager.runtime.read_pidfile", return_value=None)

    server_mod.stop_services(service="ovms", backend="docker")

    calls = mock_run.call_args_list
    stop_cmds = [c for c in calls if "stop" in c[0][0]]
    all_args = []
    for c in stop_cmds:
        all_args.extend(c[0][0])
    assert "ovmgr-server" in all_args
    assert "ovmgr-webui" not in all_args


@pytest.mark.unit
def test_stop_selective_webui_only(mocker):
    """Server stop --service webui only stops ovmgr-webui."""
    mock_run = mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    server_mod.stop_services(service="webui", backend="docker")

    calls = mock_run.call_args_list
    all_args = []
    for c in calls:
        all_args.extend(c[0][0])
    assert "ovmgr-webui" in all_args
    assert "ovmgr-server" not in all_args


@pytest.mark.unit
def test_stop_baremetal_reads_pidfile(mocker, tmp_path):
    """Baremetal stop reads PID from pidfile and sends SIGTERM."""
    pidfile = tmp_path / "ovmgr-server.pid"
    pidfile.write_text("12345")

    mocker.patch("ov_manager.commands.server.get_runtime_dir", return_value=tmp_path)
    mocker.patch("ov_manager.commands.server.read_pidfile", return_value=12345)
    mock_kill = mocker.patch("os.kill")
    mocker.patch("ov_manager.commands.server.remove_pidfile")
    # Also mock docker stop for webui
    mocker.patch("subprocess.run", return_value=MagicMock(returncode=0))

    server_mod.stop_services(service="ovms", backend="baremetal")

    import signal

    mock_kill.assert_called_once_with(12345, signal.SIGTERM)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_status_returns_service_info(mocker):
    """server_status returns info for both services."""

    # Mock docker inspect for running containers
    def fake_run(cmd, **kwargs):
        result = MagicMock()
        if "inspect" in cmd:
            result.returncode = 0
            result.stdout = '{"State": {"Running": true}}'
        else:
            result.returncode = 0
            result.stdout = ""
        return result

    mocker.patch("subprocess.run", side_effect=fake_run)
    mocker.patch("ov_manager.runtime.get_runtime_dir", return_value=Path("/tmp/rt"))
    mocker.patch("ov_manager.runtime.read_pidfile", return_value=None)

    statuses = server_mod.get_service_statuses(backend="docker")

    assert len(statuses) == 2
    names = [s["name"] for s in statuses]
    assert "ovmgr-server" in names
    assert "ovmgr-webui" in names


@pytest.mark.unit
def test_status_never_shows_auto_backend(mocker):
    """Service statuses always show 'baremetal' or 'docker', never 'auto'."""
    mocker.patch("subprocess.run", return_value=MagicMock(returncode=1, stdout=""))
    mocker.patch("ov_manager.runtime.get_runtime_dir", return_value=Path("/tmp/rt"))
    mocker.patch("ov_manager.runtime.read_pidfile", return_value=None)

    for backend_name in ("baremetal", "docker"):
        statuses = server_mod.get_service_statuses(backend=backend_name)
        for s in statuses:
            assert s["backend_type"] in ("baremetal", "docker"), (
                f"Expected 'baremetal' or 'docker', got {s['backend_type']!r}"
            )
