"""
Train a YOLOv8 model using a configuration-driven workflow.

The script keeps the original automation logic intact while abstracting away
dataset-specific assumptions. Provide a single YAML configuration that defines
where the dataset lives, how to obtain it, and which training hyperparameters
to use. See ``configs/sample.yaml`` for the expected schema.

Usage:
    python scripts/train.py --config configs/sample.yaml
"""

import argparse
import subprocess
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


def ensure_dataset(cfg_path: Path, dataset_cfg: Dict[str, Any]) -> Path:
    """Validate dataset availability; trigger download or preparation if needed."""
    dataset_dir = Path(dataset_cfg.get("local_dir", "data"))
    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        return data_yaml

    provider = dataset_cfg.get("provider", "local")
    scripts_dir = Path(__file__).parent

    print(f"Dataset not found at '{dataset_dir}'. Attempting automated preparation …")
    if provider == "kaggle":
        print("Invoking download_dataset.py to download and prepare the dataset …")
        subprocess.run(
            [sys.executable, str(scripts_dir / "download_dataset.py"), "--config", str(cfg_path)],
            check=True,
        )
    else:
        print("Invoking prepare_data.py to generate data.yaml …")
        subprocess.run(
            [sys.executable, str(scripts_dir / "prepare_data.py"), "--config", str(cfg_path)],
            check=True,
        )

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml still not found at '{data_yaml}'. Verify your dataset path and configuration."
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
        data_yaml = ensure_dataset(cfg_path, dataset_cfg)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Dataset preparation failed: {exc}")
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
