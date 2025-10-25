"""
Download raw datasets for the YOLO training workflow.

This script strictly focuses on fetching dataset assets. When the dataset
provider is ``kaggle`` it authenticates via the Kaggle API, downloads the
archive, and expands it into the configured ``local_dir``. Other providers are
treated as already-present datasets so no files are modified.

Usage:
    python scripts/download_dataset.py --config configs/sample.yaml
    python scripts/download_dataset.py --config configs/sample.yaml --env-file .env
"""

import argparse
import os
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
    except Exception as exc:  # pragma: no cover - handles CLI auth failures
        raise RuntimeError(
            "Failed to authenticate with Kaggle API. Ensure credentials are configured."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Kaggle dataset '{slug}' to '{dest_dir}' …")
    api.dataset_download_files(slug, path=str(dest_dir), unzip=True)

    for archive in dest_dir.glob("*.zip"):
        try:
            archive.unlink()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Download datasets required for YOLO training")
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
        print(f"Dataset downloaded to '{dataset_dir.resolve()}'.")
    else:
        print(
            f"Provider '{provider}' treated as local. Ensure the dataset already exists at '{dataset_dir.resolve()}'."
        )


if __name__ == "__main__":
    main()
