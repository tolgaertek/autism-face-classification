import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import IMAGENET_MEAN, IMAGENET_STD


class AutismFaceDataset(Dataset):
    """
    Folder-based dataset.
    Expects: root_dir/<class_name>/*.(jpg|png|jpeg)
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root = Path(root_dir)
        self.transform = transform

        if not self.root.exists():
            raise FileNotFoundError(f"Folder not found: {self.root}")

        # infer classes from subfolder names
        self.class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        self.image_paths = []
        self.labels = []
        exts = (".jpg", ".jpeg", ".png")

        for cls in self.class_names:
            class_dir = self.root / cls
            if not class_dir.exists():
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith(exts):
                    self.image_paths.append(class_dir / fname)
                    self.labels.append(self.class_to_idx[cls])

        if len(self.image_paths) == 0:
            print(f"Warning: no images found under {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = str(self.image_paths[idx])
        label = self.labels[idx]

        img = cv2.imread(path)
        if img is None:
            # return a safe dummy image if a file is corrupted
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label


def get_transforms(img_size: int = 224):
    """
    Returns (train_tf, test_tf)
    """
    train_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    test_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

    return train_tf, test_tf
