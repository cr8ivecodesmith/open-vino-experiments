"""Run LLM text generation via OpenVINO GenAI.

Configuration is read from config.toml at the project root and can be
overridden with CLI flags.

Example:
    python samples/text_generation.py "The Sun is yellow because"
    python samples/text_generation.py -d CPU -n 200 "Explain quantum computing"
"""

from __future__ import annotations

import json
import os
import sys
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
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output results as JSON (suppresses all rich formatting).",
)
def main(
    prompt: str,
    model_path: str | None,
    device: str | None,
    max_new_tokens: int | None,
    output_json: bool,
) -> None:
    """Main entry point for the CLI."""
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

    # -- show parameters (rich only) -----------------------------------------
    if not output_json:
        params_table = Table(show_header=False, box=None, padding=(0, 1))
        params_table.add_column(style="bold cyan")
        params_table.add_column()
        params_table.add_row("Model", str(resolved_model))
        params_table.add_row("Device", resolved_device)
        params_table.add_row("Max tokens", str(resolved_tokens))
        params_table.add_row("Prompt", prompt)
        console.print(Panel(params_table, title="Parameters", border_style="dim"))

    # -- load pipeline ------------------------------------------------------
    if output_json:
        pipe = ov_genai.LLMPipeline(str(resolved_model), resolved_device)
    else:
        with console.status(
            f"Loading model on [bold]{resolved_device}[/]...", spinner="dots"
        ):
            pipe = ov_genai.LLMPipeline(str(resolved_model), resolved_device)

    # -- generate -----------------------------------------------------------
    if output_json:
        result = pipe.generate([prompt], max_new_tokens=resolved_tokens)
    else:
        with console.status("Generating...", spinner="dots"):
            result = pipe.generate([prompt], max_new_tokens=resolved_tokens)

    m = result.perf_metrics
    total_s = m.get_generate_duration().mean / 1000

    # -- output -------------------------------------------------------------
    if output_json:
        print(
            json.dumps(
                {
                    "parameters": {
                        "model": str(resolved_model),
                        "device": resolved_device,
                        "max_new_tokens": resolved_tokens,
                        "prompt": prompt,
                    },
                    "output": result.texts[0],
                    "metrics": {
                        "tokens_in": m.get_num_input_tokens(),
                        "tokens_out": m.get_num_generated_tokens(),
                        "throughput_toks": round(m.get_throughput().mean, 1),
                        "ttft_ms": round(m.get_ttft().mean, 1),
                        "total_s": round(total_s, 2),
                    },
                },
                indent=2,
            )
        )
    else:
        console.print(
            Panel(result.texts[0], title="Generated Text", border_style="green"),
        )
        console.print(
            f"[dim]"
            f"{m.get_num_input_tokens()} in · "
            f"{m.get_num_generated_tokens()} out · "
            f"{m.get_throughput().mean:.1f} tok/s · "
            f"ttft {m.get_ttft().mean:.0f}ms · "
            f"{_humanize_duration(total_s)}"
            f"[/]"
        )


if __name__ == "__main__":
    main()
