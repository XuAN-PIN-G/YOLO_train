"""
Download and prepare datasets for the YOLO_train pipeline.

The script reads a configuration file that declares how to obtain a dataset.
When the provider is set to "kaggle", it authenticates via Kaggle API, downloads
the archive, expands it into the configured directory, optionally reshapes the
layout, and finally writes a Ultralytics-compatible ``data.yaml`` file.

Usage:
    python scripts/download_dataset.py --config configs/sample.yaml
    python scripts/download_dataset.py --config configs/sample.yaml --env-file .env
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_env_from_file(env_path: Path) -> None:
    """Populate environment variables from a simple KEY=VALUE file."""
    if not env_path.exists():
        return

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Failed to read env file '{env_path}': {exc}") from exc

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            os.environ.setdefault(key, value)


def download_kaggle_dataset(slug: str, dest_dir: Path) -> None:
    """Download and extract a Kaggle dataset into dest_dir."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle API is not installed. Add it via 'pip install kaggle' or requirements.txt."
        ) from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Failed to authenticate with Kaggle API. Ensure credentials are configured."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset '{slug}' to '{dest_dir}' …")
    api.dataset_download_files(slug, path=str(dest_dir), unzip=True)

    # Remove any leftover archives to keep the directory tidy.
    for archive in dest_dir.glob("*.zip"):
        try:
            archive.unlink()
        except OSError:
            pass


def auto_format_if_needed(dataset_dir: Path) -> None:
    """Invoke the helper script to reshape datasets that are not in YOLO format."""
    train_images = dataset_dir / "train" / "images"
    val_images = dataset_dir / "val" / "images"
    test_images = dataset_dir / "test" / "images"

    if train_images.exists() and (val_images.exists() or test_images.exists()):
        return

    print("Dataset layout not in YOLO format. Attempting automatic formatting …")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "auto_format_dataset.py"),
            "--base-dir",
            str(dataset_dir),
        ],
        check=True,
    )


def generate_data_yaml(dataset_cfg: Dict[str, Any], dataset_dir: Path) -> Path:
    """Generate Ultralytics data.yaml from configuration meta."""
    classes = dataset_cfg.get("classes")
    if not classes:
        raise ValueError("No classes declared in configuration. Populate dataset.classes.")

    auto_format_if_needed(dataset_dir)

    train_dir = dataset_dir / "train" / "images"
    val_dir = dataset_dir / "val" / "images"
    test_dir = dataset_dir / "test" / "images"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory '{train_dir}' not found after formatting. Inspect your dataset."
        )

    if val_dir.exists():
        val_images = val_dir
    elif test_dir.exists():
        val_images = test_dir
        print("Warning: 'val' split not found. Using 'test/images' for validation.")
    else:
        val_images = train_dir
        print("Warning: validation split missing. Reusing training images as fallback.")

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
    parser = argparse.ArgumentParser(description="Download and prepare datasets for YOLO training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sample.yaml",
        help="Path to the configuration YAML file",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Optional file providing Kaggle credentials (defaults to project .env).",
    )
    args = parser.parse_args()

    env_file = Path(args.env_file).expanduser() if args.env_file else DEFAULT_ENV_FILE
    load_env_from_file(env_file)

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Configuration file '{cfg_path}' not found.")
        sys.exit(1)

    cfg = load_config(cfg_path)
    dataset_cfg = cfg.get("dataset", {})
    provider = dataset_cfg.get("provider", "local")
    slug = dataset_cfg.get("slug")
    dataset_dir = Path(dataset_cfg.get("local_dir", "data"))

    if provider == "kaggle":
        if not slug:
            print("Configuration missing dataset.slug for Kaggle provider.")
            sys.exit(1)
        try:
            download_kaggle_dataset(slug, dataset_dir)
        except RuntimeError as exc:
            print(str(exc))
            sys.exit(1)
    else:
        print(
            f"Provider '{provider}' is treated as local. Ensure the dataset is already present at '{dataset_dir}'."
        )

    try:
        data_yaml = generate_data_yaml(dataset_cfg, dataset_dir)
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Generated data.yaml at {data_yaml}")


if __name__ == "__main__":
    main()
