"""
End-to-end helper that runs the YOLO training workflow in discrete stages.

The orchestrator wires together the individual scripts so each component stays
focused on its own concern (download → format → prepare → train). Each step can
be skipped via CLI switches to accommodate partially prepared datasets.

Usage:
    python scripts/run_pipeline.py --config configs/sample.yaml
    python scripts/run_pipeline.py --config configs/sample.yaml --env-file .env --skip-train
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_step(description: str, command: Iterable[str]) -> None:
    """Print a friendly banner and execute the provided command."""
    print(f"\n==> {description}")
    subprocess.run(list(command), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full YOLO dataset -> training pipeline")
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
        help="Optional env file passed to download_dataset.py (falls back to its default).",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip the download stage.")
    parser.add_argument("--skip-format", action="store_true", help="Skip auto-formatting.")
    parser.add_argument("--skip-prepare", action="store_true", help="Skip data.yaml generation.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training stage.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        print(f"Configuration file '{cfg_path}' not found.")
        sys.exit(1)

    cfg = load_config(cfg_path)
    dataset_cfg = cfg.get("dataset", {})
    dataset_dir = Path(dataset_cfg.get("local_dir", "data"))
    scripts_dir = Path(__file__).parent
    python = sys.executable

    try:
        if not args.skip_download:
            download_cmd: List[str] = [
                python,
                str(scripts_dir / "download_dataset.py"),
                "--config",
                str(cfg_path),
            ]
            if args.env_file:
                download_cmd.extend(["--env-file", str(Path(args.env_file).expanduser())])
            run_step("Downloading dataset", download_cmd)
        else:
            print("\n==> Skipping dataset download as requested.")

        if not args.skip_format:
            run_step(
                "Formatting dataset into YOLO layout",
                [
                    python,
                    str(scripts_dir / "auto_format_dataset.py"),
                    "--base-dir",
                    str(dataset_dir),
                ],
            )
        else:
            print("\n==> Skipping auto-formatting as requested.")

        if not args.skip_prepare:
            run_step(
                "Generating data.yaml",
                [
                    python,
                    str(scripts_dir / "prepare_data.py"),
                    "--config",
                    str(cfg_path),
                ],
            )
        else:
            print("\n==> Skipping data.yaml preparation as requested.")

        if not args.skip_train:
            run_step(
                "Starting training",
                [
                    python,
                    str(scripts_dir / "train.py"),
                    "--config",
                    str(cfg_path),
                ],
            )
        else:
            print("\n==> Skipping training as requested.")
    except subprocess.CalledProcessError as exc:
        print(f"\nPipeline failed during step '{exc.cmd}': {exc}")
        sys.exit(exc.returncode if exc.returncode is not None else 1)


if __name__ == "__main__":
    main()
