"""
Train a YOLOv8 model using a configuration-driven workflow.

This script is dedicated to the training stage. It assumes that the dataset
directory already contains a ``data.yaml`` file prepared by ``prepare_data.py``
or by the orchestration pipeline.

Usage:
    python scripts/train.py --config configs/sample.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError(
        "Ultralytics package not found. Install dependencies with 'pip install -r requirements.txt'."
    ) from exc


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_yaml(dataset_cfg: Dict[str, Any]) -> Path:
    """Locate the prepared data.yaml file before training."""
    dataset_dir = Path(dataset_cfg.get("local_dir", "data"))
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"data.yaml not found at '{data_yaml}'. "
            "Run scripts/prepare_data.py or the pipeline script before training."
        )
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model using a configuration file")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sample.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Configuration file '{cfg_path}' not found.")
        sys.exit(1)

    cfg = load_config(cfg_path)
    dataset_cfg = cfg.get("dataset", {})
    training_cfg = cfg.get("training", {})

    try:
        data_yaml = resolve_data_yaml(dataset_cfg)
    except FileNotFoundError as exc:
        print(f"Dataset preparation incomplete: {exc}")
        sys.exit(1)

    model_name = training_cfg.get("model", "yolov8s.pt")
    imgsz = int(training_cfg.get("imgsz", 640))
    batch = int(training_cfg.get("batch", 8))
    epochs = int(training_cfg.get("epochs", 50))
    device = training_cfg.get("device", "cpu")

    print(f"Loading model '{model_name}' …")
    model = YOLO(model_name)

    print(
        f"Starting training with parameters: imgsz={imgsz}, batch={batch}, epochs={epochs}, device={device}"
    )
    model.train(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        epochs=epochs,
        device=device,
    )
    print("Training completed.")


if __name__ == "__main__":
    main()
