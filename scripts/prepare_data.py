"""
Generate a YOLO-compatible ``data.yaml`` file from a dataset configuration.

The script inspects the dataset directory defined in the YAML configuration and
produces the metadata file required by Ultralytics. It mirrors the logic used
by the training workflow so it can be executed independently when datasets are
managed manually.

Usage:
    python scripts/prepare_data.py --config configs/sample.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_data_yaml(dataset_cfg: Dict[str, Any], dataset_dir: Path) -> Path:
    """Create a YOLO data.yaml file inside dataset_dir."""
    classes = dataset_cfg.get("classes")
    if not classes:
        raise ValueError("No classes defined in configuration. Populate dataset.classes.")

    train_dir = dataset_dir / "train" / "images"
    val_dir = dataset_dir / "val" / "images"
    test_dir = dataset_dir / "test" / "images"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory '{train_dir}' not found.")

    if val_dir.exists():
        val_images = val_dir
    elif test_dir.exists():
        val_images = test_dir
        print("Warning: 'val' directory missing. Using 'test/images' as validation split.")
    else:
        val_images = train_dir
        print("Warning: validation directory not found. Reusing training images for validation.")

    data = {
        "train": str(train_dir.resolve()),
        "val": str(val_images.resolve()),
        "nc": len(classes),
        "names": classes,
    }

    out_path = dataset_dir / "data.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a YOLO data.yaml from a dataset configuration")
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
    dataset_dir = Path(dataset_cfg.get("local_dir", "data"))

    try:
        out_yaml = generate_data_yaml(dataset_cfg, dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Created {out_yaml}")


if __name__ == "__main__":
    main()
