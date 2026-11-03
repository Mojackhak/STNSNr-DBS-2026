"""mer.nex_convert

Main purpose
------------
Convert Neuro Omega intraoperative recording files (``.map`` / ``.mpx``) into
NeuroExplorer-compatible ``.nex`` files.

Notes
-----
This conversion relies on the NeuroExplorer Python API module named ``nex``.
That module is typically available only on machines with NeuroExplorer installed.

The functions in this module are written to be import-safe even when ``nex`` is
missing; in that case, calling the conversion functions raises an ImportError
with actionable installation guidance.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .io_utils import ensure_dir, list_files

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_EXTENSIONS = (".map", ".mpx")


def _require_nex() -> object:
    """Import and return the NeuroExplorer ``nex`` module.

    Raises
    ------
    ImportError
        If the ``nex`` module cannot be imported.
    """
    try:
        import nex  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "The 'nex' module is required for MAP/MPX → NEX conversion. "
            "It usually comes with NeuroExplorer's Python API on a machine "
            "where NeuroExplorer is installed.\n\n"
            "Troubleshooting steps:\n"
            "1) Install NeuroExplorer on this machine.\n"
            "2) Ensure NeuroExplorer's Python bindings are on PYTHONPATH.\n"
            "3) Try importing 'nex' in a plain Python session.\n"
        ) from exc
    return nex


def convert_one_map_mpx_to_nex(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Convert a single ``.map``/``.mpx`` file to a ``.nex`` file.

    Parameters
    ----------
    input_path:
        Path to an input file ending with ``.map`` or ``.mpx``.
    output_path:
        Destination path ending with ``.nex``.
    overwrite:
        If False (default), keep an existing output file and return its path.

    Returns
    -------
    pathlib.Path
        The written or reused ``.nex`` path.
    """

    in_path = Path(input_path)
    out_path = Path(output_path)

    if in_path.suffix.lower() not in SUPPORTED_INPUT_EXTENSIONS:
        raise ValueError(
            f"Unsupported input extension '{in_path.suffix}'. "
            f"Expected one of {SUPPORTED_INPUT_EXTENSIONS}."
        )
    if out_path.suffix.lower() != ".nex":
        raise ValueError("output_path must end with '.nex'")

    if out_path.exists() and not overwrite:
        logger.info("Skipping existing NEX file (overwrite=False): %s", out_path)
        return out_path

    ensure_dir(out_path.parent)
    nex = _require_nex()

    doc = None
    try:
        logger.info("Converting %s -> %s", in_path, out_path)
        doc = nex.OpenDocument(str(in_path))
        nex.SaveDocumentAs(doc, str(out_path))
    finally:
        # Ensure handles are released even if conversion fails.
        if doc is not None:
            try:
                nex.CloseDocument(doc)
            except Exception:
                logger.exception("Failed to close NeuroExplorer document: %s", in_path)

    return out_path


def convert_map_mpx_to_nex(
    input_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    overwrite: bool = False,
) -> list[Path]:
    """Batch-convert all ``.map``/``.mpx`` files in a folder to ``.nex``.

    Parameters
    ----------
    input_dir:
        Folder containing ``.map`` and/or ``.mpx`` files.
    output_dir:
        Where to write ``.nex`` outputs. If None, a sibling folder named
        ``converted`` will be created inside ``input_dir``.
    recursive:
        If True, search recursively under ``input_dir``.
    overwrite:
        If True, overwrite existing ``.nex`` files. If False, existing
        outputs are kept and returned unchanged.

    Returns
    -------
    list[pathlib.Path]
        Output ``.nex`` paths.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir) if output_dir is not None else in_dir / "converted"
    ensure_dir(out_dir)

    inputs = list_files(in_dir, extensions=SUPPORTED_INPUT_EXTENSIONS, recursive=recursive)
    if not inputs:
        logger.warning("No MAP/MPX files found under: %s", in_dir)
        return []

    outputs: list[Path] = []
    for in_path in inputs:
        out_path = out_dir / f"{in_path.stem}.nex"
        outputs.append(convert_one_map_mpx_to_nex(in_path, out_path, overwrite=overwrite))

    return outputs
