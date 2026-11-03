"""parallel.py

Small wrappers around joblib Parallel + tqdm progress display.

This module provides:
- `parallel_process_files`: basic joblib parallel map
- `parallel_process_files_tqdm`: parallel map with a progress bar

Implementation note
-------------------
Some environments do not have the optional `tqdm_joblib` dependency.
When it is unavailable, we fall back to a tiny built-in context manager
that patches joblib's BatchCompletionCallBack to update a tqdm bar.
"""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
from typing import Callable, Sequence, TypeVar

from joblib import Parallel, delayed
from tqdm.auto import tqdm

try:
    # Optional dependency (nice to have, but not required)
    from tqdm_joblib import tqdm_joblib as _tqdm_joblib  # type: ignore
except Exception:  # pragma: no cover
    _tqdm_joblib = None

T = TypeVar("T")


def parallel_process_files(
    files: Sequence[object],
    func: Callable[..., T],
    n_jobs: int = -1,
    verbose: bool = False,
    **kwargs,
) -> list[T]:
    """Run a function over files in parallel (joblib).

    Parameters
    ----------
    files:
        Iterable of file-like objects (usually str/Path).
    func:
        Callable to execute. It will be called as `func(file, **kwargs)`.
    n_jobs:
        joblib parallelism. -1 uses all cores.
    verbose:
        If True, print a short summary.
    **kwargs:
        Forwarded to `func`.

    Returns
    -------
    list
        Results collected from all jobs, in the same order as `files`.
    """
    if verbose:
        print(f"Processing {len(files)} files with {n_jobs} parallel jobs")

    results: list[T] = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(func)(file, **kwargs) for file in files
    )
    return results


def _call_silently(func: Callable[..., T], file: object, kwargs: dict) -> T:
    """Call `func(file, **kwargs)` while silencing stdout/stderr."""
    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        return func(file, **kwargs)


@contextmanager
def _fallback_tqdm_joblib(tqdm_object: tqdm):
    """Minimal tqdm-joblib bridge (used when `tqdm_joblib` is not installed)."""
    import joblib.parallel  # local import to keep module load light

    original_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(original_callback):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        tqdm_object.close()


def parallel_process_files_tqdm(
    files: Sequence[object],
    func: Callable[..., T],
    n_jobs: int = -1,
    verbose: bool = False,
    prefer: str = "processes",
    silence_workers: bool = True,
    **kwargs: Any,
) -> list[T]:
    """Run a function over files in parallel with a tqdm progress bar.

    Parameters
    ----------
    files:
        Iterable of file-like objects (usually str/Path).
    func:
        Callable to execute. It will be called as `func(file, **kwargs)`.
    n_jobs:
        joblib parallelism. -1 uses all cores.
    verbose:
        If True, show progress bar and a short summary.
    prefer:
        joblib backend preference. Common values: "processes" or "threads".
    silence_workers:
        If True (default), suppress worker stdout/stderr to keep console clean.
        If False, allow worker prints to go through (may interleave across jobs).        
    **kwargs:
        Forwarded to `func`.

    Returns
    -------
    list
        Results collected from all jobs, in the same order as `files`.
    """
    if verbose:
        print(f"Processing {len(files)} files with {n_jobs} parallel jobs")

    jl_verbose = 10 if verbose else 0
    bar = tqdm(total=len(files), disable=not verbose)

    ctx = _tqdm_joblib(bar) if _tqdm_joblib is not None else _fallback_tqdm_joblib(bar)
    with ctx as _:
        if silence_workers:
            results: list[T] = Parallel(n_jobs=n_jobs, verbose=jl_verbose, prefer=prefer)(
                delayed(_call_silently)(func, file, kwargs) for file in files
            )
        else:
            # Direct call: allows worker prints, but output may interleave.
            results = Parallel(n_jobs=n_jobs, verbose=jl_verbose, prefer=prefer)(
                delayed(func)(file, **kwargs) for file in files
            )

    return results
