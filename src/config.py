from dataclasses import dataclass

# ImageNet normalization (standard)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

@dataclass
class TrainingConfig:
    # --- Model ---
    model_name: str = "resnet50"

    # --- Training hyperparameters ---
    num_classes: int = 2
    img_size: int = 224
    batch_size: int = 32
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_folds: int = 5
    seed: int = 42

    # DataLoader
    num_workers: int = 0  # keep 0 for max compatibility (Windows/Colab)

    # Optional fine-tuning control
    freeze_backbone: bool = False

    # --- Paths (LOCAL-FIRST; no Colab hardcode) ---
    # Expected: data_dir/fold_1/{train,val}/<class_name>/*.jpg
    data_dir: str = "./data/autism_unified_kfold"

    # Checkpoints (per fold)
    save_dir: str = "./saved_models"

    # Logs / figures / metrics
    output_dir: str = "./outputs"
