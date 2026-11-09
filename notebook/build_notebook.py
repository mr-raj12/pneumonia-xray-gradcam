"""Convert pneumonia_kaggle.py (`# %%` cell format) into pneumonia_kaggle.ipynb."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "pneumonia_kaggle.py"
DST = HERE / "pneumonia_kaggle.ipynb"


def to_cell(header: str, body: str) -> dict:
    lines = body.strip("\n").splitlines()
    if "[markdown]" in header:
        lines = [re.sub(r"^# ?", "", ln) for ln in lines]
        return {"cell_type": "markdown", "metadata": {}, "source": _join(lines)}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _join(lines)}


def _join(lines):
    return [ln + "\n" for ln in lines[:-1]] + lines[-1:]


def main():
    text = SRC.read_text(encoding="utf-8")
    parts = re.split(r"^(# %%.*)$", text, flags=re.MULTILINE)
    cells = [to_cell(parts[i], parts[i + 1]) for i in range(1, len(parts), 2)]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DST.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Wrote {DST} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
