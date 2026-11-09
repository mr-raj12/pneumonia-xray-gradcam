# %% [markdown]
# # Pneumonia Detection from Chest X-Rays with Grad-CAM Explainability
#
# **Goal:** classify chest X-rays as **NORMAL** or **PNEUMONIA** and show *where* the model looks using Grad-CAM.
#
# **Pipeline**
# 1. Locate dataset and re-split train/val **by patient** (the official val set has only 16 images, and a random split would leak the same patient into both sets).
# 2. Fine-tune an ImageNet-pretrained **EfficientNet-B0** (transfer learning) with class-weighted loss for the 3:1 class imbalance.
# 3. Evaluate on the untouched official **test set**: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
# 4. Explain predictions with **Grad-CAM** heatmaps.
#
# Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) - Kermany et al., 2018.

# %% [markdown]
# ## 1. Setup

# %%
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# Defaults are for Kaggle; environment variables only exist so the notebook can be smoke-tested locally.
DATA_BASE = Path(os.environ.get("DATA_BASE", "/kaggle/input"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/outputs"))
EPOCHS = int(os.environ.get("EPOCHS", 15))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
BATCH_SIZE = 32
IMG_SIZE = 224
LR = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 4          # early stopping on validation AUC
VAL_FRACTION = 0.15
SEED = 42
CLASSES = ["NORMAL", "PNEUMONIA"]

OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
print("PyTorch", torch.__version__, "| device:", DEVICE,
      f"({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "- WARNING: no GPU, enable it in Settings > Accelerator")

# %% [markdown]
# ## 2. Locate the dataset and build patient-level splits
#
# The Kaggle archive contains the data twice (`chest_xray/` and a nested `chest_xray/chest_xray/`) plus `__MACOSX` junk, so we pick the shallowest folder that has `train/` and `test/`.
#
# File names encode the patient: `person123_bacteria_456.jpeg` (pneumonia) or `IM-0115-0001.jpeg` / `NORMAL2-IM-1427-0001.jpeg` (normal). We group by that ID so no patient appears in both train and val.

# %%
def find_data_root(base: Path) -> Path:
    candidates = [p.parent for p in base.rglob("train")
                  if p.is_dir() and (p.parent / "test").is_dir() and "__MACOSX" not in p.parts]
    if not candidates:
        raise FileNotFoundError(f"No train/ + test/ folders under {base}. Did you add the dataset via 'Add Input'?")
    return min(candidates, key=lambda p: len(p.parts))

PATIENT_RE = re.compile(r"^(person\d+|NORMAL2-IM-\d+|IM-\d+)", re.IGNORECASE)

def patient_id(filename: str) -> str:
    m = PATIENT_RE.match(filename)
    return m.group(1).lower() if m else Path(filename).stem

def list_images(folder: Path) -> pd.DataFrame:
    rows = []
    for label, cls in enumerate(CLASSES):
        for p in sorted((folder / cls).glob("*")):
            if p.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                rows.append({"path": str(p), "label": label, "patient": patient_id(p.name)})
    return pd.DataFrame(rows)

DATA_ROOT = find_data_root(DATA_BASE)
print("Dataset root:", DATA_ROOT)

# Official train + tiny official val are pooled, then re-split by patient within each class (keeps class ratio).
pool = pd.concat([list_images(DATA_ROOT / "train"), list_images(DATA_ROOT / "val")], ignore_index=True)
test_df = list_images(DATA_ROOT / "test")

train_parts, val_parts = [], []
for label in range(len(CLASSES)):
    sub = pool[pool.label == label]
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    tr_idx, va_idx = next(gss.split(sub, groups=sub.patient))
    train_parts.append(sub.iloc[tr_idx])
    val_parts.append(sub.iloc[va_idx])
train_df = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = pd.concat(val_parts).reset_index(drop=True)

assert not set(train_df.patient) & set(val_df.patient), "patient leakage between train and val"

summary = pd.DataFrame({name: df.label.value_counts().reindex(range(2), fill_value=0).values
                        for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]},
                       index=CLASSES)
summary.loc["TOTAL"] = summary.sum()
print(summary)
print(f"Patients - train: {train_df.patient.nunique()}, val: {val_df.patient.nunique()} (no overlap)")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
for ax, (name, df) in zip(axes, [("Train", train_df), ("Validation", val_df), ("Test", test_df)]):
    counts = df.label.value_counts().reindex(range(2), fill_value=0)
    bars = ax.bar(CLASSES, counts.values, color=["#4c9be8", "#e8704c"])
    ax.bar_label(bars)
    ax.set_title(f"{name} ({len(df)} images)")
fig.suptitle("Class distribution")
fig.tight_layout()
fig.savefig(OUT_DIR / "class_distribution.png", dpi=150)
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for row, label in enumerate(range(2)):
    samples = train_df[train_df.label == label].sample(4, random_state=SEED)
    for ax, path in zip(axes[row], samples.path):
        ax.imshow(Image.open(path).convert("L"), cmap="gray")
        ax.set_title(CLASSES[label])
        ax.axis("off")
fig.suptitle("Sample training images")
fig.tight_layout()
fig.savefig(OUT_DIR / "sample_images.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Data loaders
#
# X-rays are grayscale; we replicate to 3 channels so ImageNet weights apply. Augmentation is kept mild (small crops, ±10° rotation, brightness/contrast) because X-rays are roughly aligned and horizontal flips would swap heart side.

# %%
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class XRayDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df.path.tolist()
        self.labels = df.label.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("L")
        return self.transform(img), self.labels[idx]

loader_kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda",
                 persistent_workers=NUM_WORKERS > 0)
train_loader = DataLoader(XRayDataset(train_df, train_tf), shuffle=True, drop_last=True, **loader_kw)
val_loader = DataLoader(XRayDataset(val_df, eval_tf), shuffle=False, **loader_kw)
test_loader = DataLoader(XRayDataset(test_df, eval_tf), shuffle=False, **loader_kw)
print(f"batches - train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")

# %% [markdown]
# ## 4. Model: EfficientNet-B0 (transfer learning)
#
# ImageNet-pretrained backbone, new 2-class head with dropout. The whole network is fine-tuned with a small learning rate.
# Class weights `w_c = N / (2 * N_c)` make the rarer NORMAL class count ~3x more in the loss.

# %%
def build_model(pretrained=True):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(model.classifier[1].in_features, len(CLASSES)))
    return model

model = build_model().to(DEVICE)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

counts = np.bincount(train_df.label, minlength=2)
class_weights = torch.tensor(len(train_df) / (2 * counts), dtype=torch.float32, device=DEVICE)
print("Class weights:", dict(zip(CLASSES, class_weights.cpu().numpy().round(3))))

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
use_amp = DEVICE.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

# %% [markdown]
# ## 5. Training
#
# Mixed precision (AMP) for speed, cosine LR schedule, and early stopping on **validation ROC-AUC**. The best checkpoint is saved.

# %%
@torch.no_grad()
def predict(model, loader):
    """Returns (true labels, P(pneumonia), mean loss)."""
    model.eval()
    ys, ps, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            logits = model(x)
        total_loss += criterion(logits.float(), y).item() * len(y)
        ps.append(logits.float().softmax(1)[:, 1].cpu())
        ys.append(y.cpu())
    y, p = torch.cat(ys).numpy(), torch.cat(ps).numpy()
    return y, p, total_loss / len(y)

def safe_auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")

CKPT = OUT_DIR / "best_model.pt"
history, best_auc, bad_epochs = [], -1.0, 0

for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    run_loss, correct, seen = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    scheduler.step()

    yv, pv, val_loss = predict(model, val_loader)
    val_acc = accuracy_score(yv, pv > 0.5)
    val_auc = safe_auc(yv, pv)
    history.append({"epoch": epoch, "train_loss": run_loss / seen, "train_acc": correct / seen,
                    "val_loss": val_loss, "val_acc": val_acc, "val_auc": val_auc})

    improved = val_auc > best_auc
    if improved:
        best_auc, bad_epochs = val_auc, 0
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_auc": float(val_auc)}, CKPT)
    else:
        bad_epochs += 1
    print(f"epoch {epoch:2d}/{EPOCHS} | train loss {run_loss / seen:.4f} acc {correct / seen:.4f} | "
          f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f} | {time.time() - t0:.0f}s"
          + ("  * saved" if improved else ""))
    if bad_epochs >= PATIENCE:
        print(f"Early stopping: no val AUC improvement for {PATIENCE} epochs.")
        break

hist = pd.DataFrame(history)
hist.to_csv(OUT_DIR / "history.csv", index=False)
print(f"Best val AUC {best_auc:.4f}")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(hist.epoch, hist.train_loss, "o-", label="train")
axes[0].plot(hist.epoch, hist.val_loss, "o-", label="validation")
axes[0].set_title("Loss")
axes[1].plot(hist.epoch, hist.train_acc, "o-", label="train")
axes[1].plot(hist.epoch, hist.val_acc, "o-", label="validation")
axes[1].set_title("Accuracy")
axes[2].plot(hist.epoch, hist.val_auc, "o-", color="tab:green", label="validation")
axes[2].set_title("ROC-AUC")
for ax in axes:
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "training_curves.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Evaluation on the held-out test set
#
# We load the best checkpoint. Besides the default 0.5 cut-off we also pick a decision threshold on the **validation** set (Youden's J = sensitivity + specificity − 1), then apply it unchanged to test - the test set is never used for any choice.

# %%
state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
model.load_state_dict(state["model"])
print(f"Loaded best checkpoint from epoch {state['epoch']} (val AUC {state['val_auc']:.4f})")

yv, pv, _ = predict(model, val_loader)
fpr_v, tpr_v, thr_v = roc_curve(yv, pv)
threshold = float(np.clip(thr_v[np.argmax(tpr_v - fpr_v)], 0.01, 0.99))
print(f"Validation-tuned threshold: {threshold:.3f}")

yt, pt, test_loss = predict(model, test_loader)

def metrics_at(th):
    pred = (pt >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
    return {"threshold": round(th, 4),
            "accuracy": accuracy_score(yt, pred),
            "precision": precision_score(yt, pred, zero_division=0),
            "recall_sensitivity": recall_score(yt, pred, zero_division=0),
            "specificity": tn / (tn + fp) if tn + fp else float("nan"),
            "f1": f1_score(yt, pred, zero_division=0)}

results = {"test_auc": safe_auc(yt, pt), "test_loss": test_loss,
           "at_0.5": metrics_at(0.5), "at_tuned": metrics_at(threshold),
           "best_epoch": int(state["epoch"]), "best_val_auc": float(state["val_auc"]),
           "n_test": int(len(yt))}
results = json.loads(json.dumps(results, default=float))
with open(OUT_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Test ROC-AUC: {results['test_auc']:.4f}\n")
print(pd.DataFrame({"threshold 0.5": results["at_0.5"], "tuned threshold": results["at_tuned"]}).round(4))

report = classification_report(yt, (pt >= threshold).astype(int), target_names=CLASSES, digits=4, zero_division=0)
(OUT_DIR / "classification_report.txt").write_text(report)
print("\nClassification report (tuned threshold):\n" + report)

# Bundle the threshold with the weights so the demo app uses the same decision rule.
torch.save({**state, "threshold": threshold, "classes": CLASSES, "img_size": IMG_SIZE}, CKPT)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, th, title in [(axes[0], 0.5, "threshold 0.5"), (axes[1], threshold, f"tuned threshold {threshold:.2f}")]:
    cm = confusion_matrix(yt, (pt >= th).astype(int), labels=[0, 1])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.1%})", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], CLASSES)
    ax.set_yticks([0, 1], CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({title})")

fpr, tpr, _ = roc_curve(yt, pt)
axes[2].plot(fpr, tpr, lw=2, label=f"EfficientNet-B0 (AUC = {results['test_auc']:.3f})")
axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="chance")
axes[2].set_xlabel("False positive rate (1 - specificity)")
axes[2].set_ylabel("True positive rate (sensitivity)")
axes[2].set_title("ROC curve - test set")
axes[2].legend(loc="lower right")
axes[2].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix_roc.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Grad-CAM explainability
#
# Grad-CAM weights the last convolutional feature maps by the gradient of the predicted class score, highlighting the regions that drove the decision. For pneumonia we expect attention on lung opacities, not on text markers or the image border.

# %%
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.handle = target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.activations = out.detach()
        out.register_hook(lambda g: setattr(self, "gradients", g.detach()))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        probs = logits.softmax(1)[0].detach()
        class_idx = int(probs.argmax()) if class_idx is None else class_idx
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.cpu().numpy(), probs.cpu().numpy()

def overlay(img, cam, alpha=0.45):
    return np.clip((1 - alpha) * img + alpha * colormaps["jet"](cam)[..., :3], 0, 1)

model.eval()
cam_fn = GradCAM(model, model.features[-1])
pred_t = (pt >= threshold).astype(int)
rng = np.random.default_rng(SEED)
picks = []
for label in range(2):  # 3 correct per class + up to 2 mistakes
    correct_idx = np.where((yt == label) & (pred_t == label))[0]
    picks += [(i, "correct") for i in rng.choice(correct_idx, min(3, len(correct_idx)), replace=False)]
wrong_idx = np.where(yt != pred_t)[0]
picks += [(i, "WRONG") for i in rng.choice(wrong_idx, min(2, len(wrong_idx)), replace=False)]

fig, axes = plt.subplots(2, len(picks), figsize=(2.8 * len(picks), 6))
axes = np.array(axes).reshape(2, -1)
for col, (i, tag) in enumerate(picks):
    raw = Image.open(test_df.path[i]).convert("L")
    x = eval_tf(raw).unsqueeze(0).to(DEVICE)
    cam, probs = cam_fn(x)
    img = np.repeat(np.asarray(raw.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)[..., None] / 255.0, 3, axis=2)
    axes[0, col].imshow(img)
    axes[0, col].set_title(f"true: {CLASSES[yt[i]]}", fontsize=9)
    axes[1, col].imshow(overlay(img, cam))
    axes[1, col].set_title(f"pred: {CLASSES[pred_t[i]]} ({pt[i]:.2f})\n{tag}", fontsize=9,
                           color="green" if tag == "correct" else "red")
    for ax in axes[:, col]:
        ax.axis("off")
cam_fn.handle.remove()
fig.suptitle("Grad-CAM on test images (top: input, bottom: heatmap; number = P(pneumonia))")
fig.tight_layout()
fig.savefig(OUT_DIR / "gradcam_samples.png", dpi=150)
plt.show()

# %% [markdown]
# ## 8. Package outputs for download
#
# Copies a few test X-rays for the demo app, then zips everything into `/kaggle/working/pneumonia_outputs.zip` (download it from the **Output** panel on the right).

# %%
sample_dir = OUT_DIR / "sample_xrays"
sample_dir.mkdir(exist_ok=True)
for label in range(2):
    for path in test_df[test_df.label == label].sample(3, random_state=SEED).path:
        shutil.copy(path, sample_dir / f"{CLASSES[label].lower()}_{Path(path).name}")

zip_path = shutil.make_archive(str(OUT_DIR.parent / "pneumonia_outputs"), "zip", OUT_DIR)
print("Created", zip_path)
for p in sorted(OUT_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(OUT_DIR)}  ({p.stat().st_size / 1e6:.2f} MB)")
