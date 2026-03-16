"""Run LLM text generation via OpenVINO GenAI.

Configuration is read from config.toml at the project root and can be
overridden with CLI flags.

Example:

    python samples/text_generation.py "The Sun is yellow because"
    python samples/text_generation.py -d CPU -n 200 "Explain quantum computing"
"""

from __future__ import annotations

import os
import sys
import time
import tomllib
from pathlib import Path

import click
import openvino_genai as ov_genai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DEVICE = "CPU"
DEFAULT_MAX_NEW_TOKENS = 100

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.toml"

console = Console()


def load_config() -> dict:
    """Load and return the config.toml at the project root.

    Returns an empty dict if the file does not exist.
    """
    if not _CONFIG_PATH.is_file():
        return {}
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _humanize_duration(seconds: float) -> str:
    """Return a human-friendly elapsed time string.

    Examples: "1.2s", "3m 12s", "1h 5m 30s".
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {secs}s"


def expand_path(raw: str) -> Path:
    """Expand ``~`` and ``$ENV_VAR`` references, then resolve the path.

    Raises ``click.BadParameter`` if the expanded path does not exist.
    """
    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = Path(expanded).resolve()
    if not p.exists():
        raise click.BadParameter(
            f"Path does not exist after expansion: {p}\n  (original value: {raw})"
        )
    return p


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command(
    help=(
        "Run LLM text generation via OpenVINO GenAI.\n\n"
        'Try: "The Sun is yellow because"'
    ),
)
@click.argument("prompt")
@click.option(
    "-m",
    "--model",
    "model_path",
    default=None,
    help="Path to the model directory (~ and $VAR expanded).",
)
@click.option(
    "-d",
    "--device",
    default=None,
    help="Device to run on: NPU, GPU, or CPU.",
)
@click.option(
    "-n",
    "--max-new-tokens",
    "max_new_tokens",
    type=int,
    default=None,
    help=f"Maximum number of tokens to generate [default: {DEFAULT_MAX_NEW_TOKENS}].",
)
def main(
    prompt: str,
    model_path: str | None,
    device: str | None,
    max_new_tokens: int | None,
) -> None:
    config = load_config()

    # -- resolve model path (CLI > config > error) -------------------------
    raw_model = model_path or config.get("model", {}).get("path")
    if raw_model is None:
        console.print(
            "[bold red]Error:[/] No model path provided. "
            "Set [bold]model.path[/] in config.toml or pass [bold]-m/--model[/]."
        )
        sys.exit(1)
    resolved_model = expand_path(raw_model)

    # -- resolve device (CLI > config > default) ----------------------------
    resolved_device = (
        device or config.get("device", {}).get("default") or DEFAULT_DEVICE
    )

    # -- resolve max_new_tokens (CLI > default) -----------------------------
    resolved_tokens = (
        max_new_tokens if max_new_tokens is not None else DEFAULT_MAX_NEW_TOKENS
    )

    # -- show parameters -----------------------------------------------------
    params_table = Table(show_header=False, box=None, padding=(0, 1))
    params_table.add_column(style="bold cyan")
    params_table.add_column()
    params_table.add_row("Model", str(resolved_model))
    params_table.add_row("Device", resolved_device)
    params_table.add_row("Max tokens", str(resolved_tokens))
    params_table.add_row("Prompt", prompt)
    console.print(Panel(params_table, title="Parameters", border_style="dim"))

    # -- load pipeline with a spinner ---------------------------------------
    with console.status(
        f"Loading model on [bold]{resolved_device}[/]...", spinner="dots"
    ):
        pipe = ov_genai.LLMPipeline(str(resolved_model), resolved_device)

    # -- generate -----------------------------------------------------------
    t0 = time.perf_counter()
    with console.status("Generating...", spinner="dots"):
        result = pipe.generate(prompt, max_new_tokens=resolved_tokens)
    elapsed = time.perf_counter() - t0

    console.print(Panel(result, title="Generated Text", border_style="green"))
    console.print(f"[dim]Completed in {_humanize_duration(elapsed)}[/]")


if __name__ == "__main__":
    main()
