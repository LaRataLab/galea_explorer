"""Run the notebooks top to bottom on one session, without the folder dialog.

    python tools/run_notebooks.py <session folder> [notebook ...]

Default: both notebooks. Executed copies go to the system temp folder; the notebooks in the
repo are not modified. Exits non-zero if any cell raises.
"""
from __future__ import annotations

import os
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = ["dataset_quality.ipynb", "dataset_explorer.ipynb"]


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 2
    session, notebooks = os.path.abspath(argv[0]), argv[1:] or DEFAULT
    if not os.path.isdir(session):
        print(f"not a folder: {session}")
        return 2
    os.environ["GALEA_DATA"] = session
    failed = 0
    for name in notebooks:
        nb = nbformat.read(os.path.join(REPO, name), as_version=4)
        out = os.path.join(tempfile.gettempdir(), "executed_" + os.path.basename(name))
        try:
            NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": REPO}}).execute()
            print(f"OK    {name}")
        except CellExecutionError as e:
            failed += 1
            print(f"FAIL  {name}\n{str(e)[-2000:]}")
        finally:
            nbformat.write(nb, out)
            print(f"      executed copy: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
