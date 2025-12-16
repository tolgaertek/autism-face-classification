import argparse

from src.config import TrainingConfig
from src.train import run_kfold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autism Face Classification - Physical K-Fold Training")

    parser.add_argument("--data_dir", type=str, default="./data/autism_unified_kfold", help="Path to k-fold dataset root")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="Where to save fold checkpoints")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Where to save logs/metrics")

    parser.add_argument("--model", type=str, default="resnet50", help="resnet50 | mobilenet_v3_small | densenet121")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze_backbone", action="store_true", help="If set, freeze pretrained backbone and train only head")

    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        lr=args.lr,
        weight_decay=args.wd,
        num_folds=args.folds,
        seed=args.seed,
        freeze_backbone=args.freeze_backbone,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        output_dir=args.output_dir,
    )

    run_kfold(cfg)
