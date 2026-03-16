"""Core scraping logic for the OpenVINO test-data mirror.

The site at ``storage.openvinotoolkit.org`` exposes a ``filetree.json``
manifest that describes every file and directory in the bucket.  This module
downloads that manifest once, walks the tree under ``data/test_data/``, and
mirrors every file to a local directory — with resume support, per-thread
random delay, and configurable parallelism.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, cast

import httpx
from tqdm import tqdm

from ov_testdata_scraper import state as st

BUCKET_URL = "https://storage.openvinotoolkit.org"
FILETREE_URL = f"{BUCKET_URL}/filetree.json"
ROOT_PREFIX = "data/test_data"

# Generous timeout — the filetree JSON is >5 MB and some video files are large.
CONNECT_TIMEOUT = 30.0
READ_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------


def fetch_filetree(delay: float) -> dict:
    """Download and parse the bucket file-tree manifest.

    Args:
        delay: Maximum random delay in seconds before the request.

    Returns:
        The parsed JSON tree (root node).
    """
    _random_delay(delay)
    with httpx.Client(timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT)) as client:
        resp = client.get(FILETREE_URL)
        resp.raise_for_status()
        return cast(dict, resp.json())


def walk_tree(node: dict, prefix: str = "") -> Iterator[tuple[str, int]]:
    """Recursively yield ``(relative_path, size)`` for every file under *node*.

    Directories are traversed; files are yielded with their full path built by
    joining *prefix* and the node ``name``.

    Args:
        node: A filetree node dict with keys ``name``, ``type``, ``children``, ``size``.
        prefix: The path accumulated so far (empty at the root).

    Yields:
        Tuples of ``(relative_path, file_size_in_bytes)``.
    """
    children = node.get("children", [])
    for child in children:
        name = child.get("name", "")
        child_path = f"{prefix}/{name}" if prefix else name
        if child.get("type") == "directory":
            yield from walk_tree(child, child_path)
        else:
            yield child_path, child.get("size", 0)


def _find_subtree(tree: dict, path: str) -> dict | None:
    """Navigate from *tree* root to the node at *path* (slash-separated).

    Returns ``None`` if the path cannot be found.
    """
    parts = [p for p in path.split("/") if p]
    node = tree
    for part in parts:
        children = node.get("children", [])
        found = None
        for child in children:
            if child.get("name") == part and child.get("type") == "directory":
                found = child
                break
        if found is None:
            return None
        node = found
    return node


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _random_delay(max_seconds: float) -> None:
    """Sleep for a random duration between 0 and *max_seconds*."""
    if max_seconds > 0:
        time.sleep(random.uniform(0, max_seconds))


def _download_file(
    rel_path: str,
    size: int,
    out_dir: Path,
    delay: float,
    download_state: dict[str, dict],
    state_lock: threading.Lock,
) -> str:
    """Download a single file, update state, and return *rel_path* on success.

    This function is designed to run inside a :class:`ThreadPoolExecutor`.

    Args:
        rel_path: Slash-separated path relative to the bucket root
                  (e.g. ``data/test_data/images/dog.jpg``).
        size: Expected file size in bytes (from the manifest).
        out_dir: Local root directory to mirror into.
        delay: Max random per-request delay in seconds.
        download_state: Shared mutable state dict (protected by *state_lock*).
        state_lock: Threading lock guarding *download_state* mutations and
                    persistence.

    Returns:
        The *rel_path* of the downloaded file.

    Raises:
        httpx.HTTPStatusError: If the server responds with an error status.
    """
    # Strip the ROOT_PREFIX so local paths start at the test_data contents.
    # e.g. "data/test_data/images/dog.jpg" → "images/dog.jpg"
    dest = out_dir / _strip_prefix(rel_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    url = f"{BUCKET_URL}/{rel_path}"

    _random_delay(delay)

    with httpx.Client(
        timeout=httpx.Timeout(CONNECT_TIMEOUT, read=READ_TIMEOUT),
        follow_redirects=True,
    ) as client:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=65_536):
                    fh.write(chunk)

    # Record success atomically.
    with state_lock:
        st.record_download(download_state, rel_path, size)
        st.save(out_dir, download_state)

    return rel_path


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _humanize_bytes(n: int) -> str:
    """Return a human-friendly string for *n* bytes."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} PB"


def _strip_prefix(path: str) -> str:
    """Strip ``ROOT_PREFIX`` from *path*, returning the test_data-relative portion."""
    if path.startswith(ROOT_PREFIX + "/"):
        return path[len(ROOT_PREFIX) + 1 :]
    if path.startswith(ROOT_PREFIX):
        return path[len(ROOT_PREFIX) :]
    return path


def _print_dry_run(pending: list[tuple[str, int]]) -> None:
    """Print the list of files that would be downloaded and the total size.

    Args:
        pending: List of ``(rel_path, size)`` tuples not yet downloaded.
    """
    total_size = 0
    for path, size in pending:
        tqdm.write(f"  {_strip_prefix(path)}  ({_humanize_bytes(size)})")
        total_size += size
    tqdm.write(f"\nTotal: {len(pending)} file(s), {_humanize_bytes(total_size)}")


def _run_downloads(
    pending: list[tuple[str, int]],
    out_dir: Path,
    workers: int,
    delay: float,
    download_state: dict[str, dict],
) -> None:
    """Download all *pending* files in parallel and print a summary.

    Args:
        pending: List of ``(rel_path, size)`` tuples to download.
        out_dir: Local mirror root directory.
        workers: Number of parallel download threads.
        delay: Max random per-thread delay in seconds.
        download_state: Shared state dict (mutated in-place as downloads complete).
    """
    state_lock = threading.Lock()
    errors: list[tuple[str, str]] = []

    with tqdm(total=len(pending), unit="file", desc="Downloading") as pbar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _download_file,
                    rel_path,
                    size,
                    out_dir,
                    delay,
                    download_state,
                    state_lock,
                ): rel_path
                for rel_path, size in pending
            }
            for future in as_completed(futures):
                rel_path = futures[future]
                try:
                    future.result()
                except Exception as exc:  # noqa: BLE001
                    errors.append((rel_path, str(exc)))
                    tqdm.write(f"ERROR downloading {rel_path}: {exc}")
                finally:
                    pbar.update(1)

    succeeded = len(pending) - len(errors)
    tqdm.write(f"\nDone. {succeeded}/{len(pending)} file(s) downloaded successfully.")
    if errors:
        tqdm.write(f"{len(errors)} file(s) failed (re-run to retry):")
        for path, err in errors:
            tqdm.write(f"  {path}: {err}")


def run(
    out_dir: Path,
    workers: int = 4,
    delay: float = 2.0,
    dry_run: bool = False,
    refresh: bool = False,
) -> None:
    """Execute the scraper.

    Args:
        out_dir: Local directory to mirror files into.
        workers: Number of parallel download threads.
        delay: Maximum random delay (seconds) applied per-thread before each
               HTTP request.
        dry_run: If ``True``, list files that would be downloaded but do
                 nothing.
        refresh: If ``True``, ignore prior state and re-download everything.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if refresh:
        st.clear(out_dir)

    download_state = st.load(out_dir)

    tqdm.write("Fetching file manifest from storage.openvinotoolkit.org ...")
    tree = fetch_filetree(delay)

    subtree = _find_subtree(tree, ROOT_PREFIX)
    if subtree is None:
        tqdm.write(f"ERROR: Could not locate '{ROOT_PREFIX}' in the file tree.")
        raise SystemExit(1)

    all_files: list[tuple[str, int]] = list(walk_tree(subtree, ROOT_PREFIX))
    tqdm.write(f"Found {len(all_files)} files in the manifest.")

    pending: list[tuple[str, int]] = [
        (path, size) for path, size in all_files if not st.is_downloaded(download_state, path, size)
    ]

    skipped = len(all_files) - len(pending)
    if skipped:
        tqdm.write(f"Skipping {skipped} already-downloaded file(s).")
    tqdm.write(f"{len(pending)} file(s) to download.")

    if dry_run:
        _print_dry_run(pending)
        return

    if not pending:
        tqdm.write("Nothing to download — all files are up to date.")
        return

    _run_downloads(pending, out_dir, workers, delay, download_state)
