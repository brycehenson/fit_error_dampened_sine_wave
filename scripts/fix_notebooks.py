#!/usr/bin/env python3
"""Normalize all Jupyter notebooks in the repository.

This will add missing ``id`` fields to cells and rewrite the
notebooks in-place.  nbformat.normalize() became available in
5.1.4 and higher; prior versions fixed the issue silently but the
warning above shows that future versions will raise a hard error.

Usage::

    python scripts/fix_notebooks.py

It walks the workspace looking for ``*.ipynb`` files and ensures every
cell has an ``id``.  You can run this manually or hook it into a
pre-commit configuration (see README tweaks below).
"""

import glob
import sys

import nbformat


def normalize_notebook(path: str) -> None:
    """Read, normalize and overwrite notebook at ``path``.

    This uses ``nbformat.validator.normalize`` for recent versions and
    falls back to a minimal id-generation pass for earlier releases where
    ``normalize`` isn't available.
    """
    nb = nbformat.read(path, as_version=4)
    # ``nb`` is a NotebookNode which behaves like a dict; the validator
    # wants a dict-like object.
    try:
        from nbformat.validator import normalize

        changes, new_nb = normalize(nb)
        # ``new_nb`` is a deep copy; write that back so any added ids stick
        nb = new_nb
    except ImportError:  # pragma: no cover - older nbformat
        # fallback: add ids to cells that are missing them
        for cell in nb.cells:
            if cell.get("id", "") == "":
                cell.id = nbformat.v4.new_code_cell().id
    nbformat.write(nb, path)
    print(f"normalized {path}")


def main() -> None:
    # search from repo root
    for path in glob.glob("**/*.ipynb", recursive=True):
        try:
            normalize_notebook(path)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"failed to process {path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
