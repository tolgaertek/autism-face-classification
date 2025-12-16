import os
import json
from typing import Optional, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .dataset import AutismFaceDataset, get_transforms
from .models import create_model
from .utils import evaluate_metrics


def evaluate_checkpoint(
    data_root: str,
    model_name: str,
    weights_path: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_classes: int = 2,
    device: Optional[str] = None,
    save_dir: str = "./outputs/eval",
    class_names: Optional[List[str]] = None,
) -> Tuple[dict, np.ndarray]:
    """
    Evaluates a single checkpoint on a folder (expects class subfolders).
    Exports:
      - metrics.json
      - confusion_matrix.png
    """
    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_tf = get_transforms(img_size)
    ds = AutismFaceDataset(data_root, transform=test_tf)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_model(model_name, num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            outs = model(imgs)
            _, preds = outs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    acc, prec, rec, f1, cm = evaluate_metrics(all_labels, all_preds)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "num_samples": int(len(ds)),
        "model_name": model_name,
        "weights_path": weights_path,
        "data_root": data_root,
    }

    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if class_names is None:
        class_names = getattr(ds, "class_names", [str(i) for i in range(cm.shape[0])])

    # confusion_matrix.png
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(cm.shape[0])
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=200)
    plt.close()

    return metrics, cm
