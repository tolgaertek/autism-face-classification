import argparse
from src.eval import evaluate_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and export metrics/figures.")
    parser.add_argument("--data", required=True, help="Evaluation folder (e.g., ./data/.../fold_1/val)")
    parser.add_argument("--model", default="resnet50", help="resnet50 | mobilenet_v3_small | densenet121")
    parser.add_argument("--weights", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", default="./outputs/eval", help="Output folder for metrics.json and confusion_matrix.png")

    args = parser.parse_args()

    metrics, _ = evaluate_checkpoint(
        data_root=args.data,
        model_name=args.model,
        weights_path=args.weights,
        img_size=args.img_size,
        batch_size=args.batch,
        save_dir=args.out,
    )

    print("Evaluation complete.")
    print(metrics)
