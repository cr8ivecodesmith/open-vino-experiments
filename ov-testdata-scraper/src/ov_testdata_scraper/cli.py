"""CLI entry point for the OpenVINO test-data scraper.

Usage::

    ov-testdata-scraper --out-dir ov-testdata
    ov-testdata-scraper --out-dir ov-testdata --workers 8 --dry-run
    ov-testdata-scraper --out-dir ov-testdata --refresh
"""

from __future__ import annotations

from pathlib import Path

import click

from ov_testdata_scraper.scraper import run


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "-o",
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("ov-testdata"),
    show_default=True,
    help="Directory to mirror downloaded test data into.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=4,
    show_default=True,
    help="Number of parallel download threads.",
)
@click.option(
    "-d",
    "--delay",
    type=float,
    default=2.0,
    show_default=True,
    help="Maximum random delay (in seconds) applied per-thread before each HTTP request.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="List files that would be downloaded without actually downloading.",
)
@click.option(
    "--refresh",
    is_flag=True,
    default=False,
    help="Re-download all files, ignoring previous progress. Prompts for confirmation.",
)
def main(
    out_dir: Path,
    workers: int,
    delay: float,
    dry_run: bool,
    refresh: bool,
) -> None:
    """Scrape OpenVINO test data from storage.openvinotoolkit.org.

    Downloads images, videos, and other test assets from the
    ``data/test_data/`` subtree of the OpenVINO storage bucket, mirroring the
    remote directory structure into --out-dir.

    Interrupted runs are automatically resumed on the next invocation — only
    files that were not yet successfully downloaded will be fetched.
    """
    if refresh and not dry_run:
        click.confirm(
            f"WARNING: --refresh will re-download ALL files into '{out_dir}'. Continue?",
            abort=True,
        )

    run(
        out_dir=out_dir,
        workers=workers,
        delay=delay,
        dry_run=dry_run,
        refresh=refresh,
    )
