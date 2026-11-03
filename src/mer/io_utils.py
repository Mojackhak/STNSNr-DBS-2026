"""mer.io_utils

Main purpose
------------
Small, dependency-light utilities for file discovery and safe path handling.

This module intentionally avoids importing heavy optional dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    pathlib.Path
        The created (or existing) directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(
    root: str | Path,
    extensions: Iterable[str] = (),
    recursive: bool = True,
) -> list[Path]:
    """List files under a directory filtered by extensions.

    Parameters
    ----------
    root:
        Folder to scan.
    extensions:
        File extensions to include (e.g., [".mat", ".mpx"]).
        Matching is case-insensitive. If empty, all files are returned.
    recursive:
        If True, recursively scan subfolders.

    Returns
    -------
    list[pathlib.Path]
        Sorted file paths.
    """
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {root_path}")

    exts = tuple(e.lower() for e in extensions)
    pattern = "**/*" if recursive else "*"

    files: list[Path] = []
    for p in root_path.glob(pattern):
        if not p.is_file():
            continue
        if exts and p.suffix.lower() not in exts:
            continue
        files.append(p)

    return sorted(files)
