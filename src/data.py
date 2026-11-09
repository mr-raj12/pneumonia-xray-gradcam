"""Dataset, transforms and split helpers for the chest X-ray pneumonia dataset."""
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"


def train_transforms():
    # X-rays are roughly aligned, so only mild geometric/intensity augmentation.
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalisation on a CHW tensor, returning values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=t.device).view(3, 1, 1)
    return (t * std + mean).clamp(0, 1)


class XRayDataset(Dataset):
    """Reads (path, label) rows from a split CSV written by prepare_data.py."""

    def __init__(self, split: str, transform=None):
        csv = SPLITS_DIR / f"{split}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"{csv} missing - run prepare_data.py first")
        self.df = pd.read_csv(csv)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.path).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, int(row.label)

    def class_counts(self):
        return self.df.label.value_counts().reindex(range(len(CLASSES)), fill_value=0).tolist()
