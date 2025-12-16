import os
import json
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .dataset import AutismFaceDataset, get_transforms
from .models import create_model
from .utils import set_seed, evaluate_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = outs.max(1)
        correct += preds.eq(lbls).sum().item()
        total += lbls.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            loss = criterion(outs, lbls)

            total_loss += loss.item() * imgs.size(0)
            _, preds = outs.max(1)
            correct += preds.eq(lbls).sum().item()
            total += lbls.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_kfold(cfg: TrainingConfig):
    """
    Physically separated K-Fold training.
    Expects: cfg.data_dir/fold_{k}/{train,val}/<class_name>/*
    """
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting: {cfg.model_name} | Physical K-Fold | device={device}")

    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_tf, test_tf = get_transforms(cfg.img_size)

    results = {"acc": [], "prec": [], "rec": [], "f1": []}
    global_cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=int)

    exp_name = f"{cfg.model_name}_kfold{cfg.num_folds}_img{cfg.img_size}_bs{cfg.batch_size}_lr{cfg.lr}"
    exp_dir = os.path.join(cfg.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # save config snapshot
    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    for fold in range(1, cfg.num_folds + 1):
        fold_dir = os.path.join(cfg.data_dir, f"fold_{fold}")
        print(f"\nFold {fold}/{cfg.num_folds} -> {fold_dir}")

        train_path = os.path.join(fold_dir, "train")
        val_path = os.path.join(fold_dir, "val")

        ds_train = AutismFaceDataset(train_path, transform=train_tf)
        ds_val = AutismFaceDataset(val_path, transform=test_tf)

        train_loader = DataLoader(
            ds_train,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        val_loader = DataLoader(
            ds_val,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        print(f"Samples -> train={len(ds_train)} | val={len(ds_val)}")

        model = create_model(cfg.model_name, cfg.num_classes, freeze_backbone=cfg.freeze_backbone).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best_loss = float("inf")
        ckpt_path = os.path.join(cfg.save_dir, f"{cfg.model_name}_fold{fold}.pth")

        history = []

        for epoch in range(cfg.num_epochs):
            tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
            vl, va = validate(model, val_loader, criterion, device)

            if vl < best_loss:
                best_loss = vl
                torch.save(model.state_dict(), ckpt_path)

            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(tl),
                    "train_acc": float(ta),
                    "val_loss": float(vl),
                    "val_acc": float(va),
                    "best_val_loss_so_far": float(best_loss),
                }
            )

            print(
                f"Epoch {epoch+1:02d}/{cfg.num_epochs} | "
                f"train_loss={tl:.4f} train_acc={ta:.4f} | "
                f"val_loss={vl:.4f} val_acc={va:.4f}"
            )

        # save per-fold history
        with open(os.path.join(exp_dir, f"fold_{fold}_history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        # Evaluate best checkpoint on val
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                outs = model(imgs)
                _, preds = outs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(lbls.numpy())

        acc, prec, rec, f1, cm = evaluate_metrics(all_labels, all_preds)

        results["acc"].append(float(acc))
        results["prec"].append(float(prec))
        results["rec"].append(float(rec))
        results["f1"].append(float(f1))
        global_cm += cm

        print(f"Fold {fold} result -> acc={acc:.4f} rec={rec:.4f} f1={f1:.4f}")

    summary = {
        "model_name": cfg.model_name,
        "acc_mean": float(np.mean(results["acc"])) if results["acc"] else None,
        "acc_std": float(np.std(results["acc"])) if results["acc"] else None,
        "prec_mean": float(np.mean(results["prec"])) if results["prec"] else None,
        "prec_std": float(np.std(results["prec"])) if results["prec"] else None,
        "rec_mean": float(np.mean(results["rec"])) if results["rec"] else None,
        "rec_std": float(np.std(results["rec"])) if results["rec"] else None,
        "f1_mean": float(np.mean(results["f1"])) if results["f1"] else None,
        "f1_std": float(np.std(results["f1"])) if results["f1"] else None,
        "global_confusion_matrix": global_cm.tolist(),
        "per_fold": results,
    }

    with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"{cfg.model_name.upper()} - PHYSICAL K-FOLD SUMMARY")
    print("=" * 60)
    print(f"Accuracy  : {summary['acc_mean']:.4f} ± {summary['acc_std']:.4f}")
    print(f"Precision : {summary['prec_mean']:.4f} ± {summary['prec_std']:.4f}")
    print(f"Recall    : {summary['rec_mean']:.4f} ± {summary['rec_std']:.4f}")
    print(f"F1        : {summary['f1_mean']:.4f} ± {summary['f1_std']:.4f}")
    print("Global CM :")
    print(global_cm)
    print(f"Saved summary -> {os.path.join(exp_dir, 'summary.json')}")
