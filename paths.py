"""Portable path configuration for the HFYC_NPS analysis package.

The public package deliberately excludes participant-level data.  Set
``HFYC_DATA`` to the tab-delimited coded extract supplied under the approved
data-use agreement before running an analysis.  Optionally set ``HFYC_DICT``
to the variable-dictionary workbook and ``HFYC_OUT`` to a writable results
directory.  Command-line options in ``analysis_cc.py`` override these
environment variables for the complete-case pipeline.
"""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def data_path(value: str | Path | None = None) -> Path:
    """Return the coded input path or raise an actionable configuration error."""

    candidate = Path(value).expanduser().resolve() if value else _path_from_env("HFYC_DATA")
    if candidate is None:
        raise FileNotFoundError(
            "HFYC_DATA is not set. Provide --data to analysis_cc.py or set "
            "HFYC_DATA to the approved tab-delimited HFYC_NPS.xls extract."
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"Configured HFYC_DATA does not exist: {candidate}")
    return candidate


def dictionary_path(value: str | Path | None = None) -> Path:
    """Return the optional variable dictionary workbook, with a clear error."""

    candidate = Path(value).expanduser().resolve() if value else _path_from_env("HFYC_DICT")
    if candidate is None:
        raise FileNotFoundError(
            "HFYC_DICT is not set. Provide --dictionary to analysis_cc.py or set "
            "HFYC_DICT to the variable-dictionary .xlsx file."
        )
    if not candidate.is_file():
        raise FileNotFoundError(f"Configured HFYC_DICT does not exist: {candidate}")
    return candidate


def output_root(value: str | Path | None = None) -> Path:
    """Return a writable output root without writing inside the source package."""

    candidate = Path(value).expanduser().resolve() if value else _path_from_env("HFYC_OUT")
    return candidate or (PACKAGE_ROOT / "results").resolve()


def output_dir(name: str, value: str | Path | None = None) -> Path:
    """Create and return a named output directory."""

    root = output_root(value)
    result = root / name
    result.mkdir(parents=True, exist_ok=True)
    return result

