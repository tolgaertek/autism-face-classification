import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def set_seed(seed: int = 42):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to: {seed}")


def evaluate_metrics(all_labels, all_preds):
    """Returns: acc, prec_macro, rec_macro, f1_macro, confusion_matrix"""
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)
    return acc, prec, rec, f1, cm
