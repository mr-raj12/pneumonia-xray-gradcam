"""Create a tiny fake copy of the Kaggle dataset layout (incl. the nested duplicate
and __MACOSX folder) so the notebook can be smoke-tested without the real 2 GB data."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

base = Path(sys.argv[1])
rng = np.random.default_rng(0)


def write_split(root: Path, split: str, n_patients: int, per_patient: int):
    for cls in ["NORMAL", "PNEUMONIA"]:
        d = root / split / cls
        d.mkdir(parents=True, exist_ok=True)
        for p in range(n_patients):
            for k in range(per_patient):
                if cls == "NORMAL":
                    name = f"IM-{p:04d}-{k:04d}.jpeg" if p % 2 else f"NORMAL2-IM-{p:04d}-{k:04d}.jpeg"
                    arr = rng.normal(90, 30, (180, 220))
                else:
                    name = f"person{p}_{'bacteria' if p % 2 else 'virus'}_{k}.jpeg"
                    arr = rng.normal(150, 30, (200, 240))  # brighter = learnable signal
                Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(d / name)


root = base / "chest-xray-pneumonia" / "chest_xray"
for r in [root, root / "chest_xray"]:  # real archive has a nested duplicate
    write_split(r, "train", n_patients=20, per_patient=3)
    write_split(r, "val", n_patients=2, per_patient=2)
    write_split(r, "test", n_patients=8, per_patient=2)
(root / "__MACOSX" / "train").mkdir(parents=True, exist_ok=True)
(root / "__MACOSX" / "test").mkdir(parents=True, exist_ok=True)
print("fake dataset at", root)
