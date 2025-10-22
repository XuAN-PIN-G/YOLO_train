"""
Reformat raw datasets into the directory layout expected by Ultralytics YOLO.

Many public datasets ship as a flat folder containing images and label files or
use custom splits. This helper scans the given directory, copies assets into a
standard ``train/`` and ``val/`` structure, and keeps labels aligned with their
corresponding images.

Usage:
    python scripts/auto_format_dataset.py --base-dir data/custom_dataset
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List

SUPPORTED_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


def discover_source_dirs(base_path: Path) -> List[Path]:
    """Locate directories containing mixed image/label files."""
    candidates: List[Path] = []
    for directory in sorted(p for p in base_path.rglob("*") if p.is_dir()):
        images = [p for p in directory.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        labels = [p for p in directory.iterdir() if p.suffix.lower() == ".txt"]
        if images and labels:
            candidates.append(directory)
    return candidates


def ensure_clean_targets(base_path: Path) -> None:
    """Make sure target directories exist and are empty to avoid duplicates."""
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            target = base_path / split / sub
            if target.exists() and any(target.iterdir()):
                raise RuntimeError(
                    f"Target directory '{target}' is not empty. Remove or back it up before reformatting."
                )
            target.mkdir(parents=True, exist_ok=True)


def copy_split(images: List[Path], base_path: Path, split: str) -> None:
    for img_path in tqdm(images, desc=f"Copying {split} files"):
        label_path = img_path.with_suffix(".txt")
        dest_img = base_path / split / "images" / img_path.name
        dest_lbl = base_path / split / "labels" / label_path.name
        shutil.copy2(img_path, dest_img)
        if label_path.exists():
            shutil.copy2(label_path, dest_lbl)


def auto_format_dataset(base_dir: str, train_split: float = 0.8, seed: int = 42) -> None:
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory '{base_path}' does not exist.")

    train_images = base_path / "train" / "images"
    val_images = base_path / "val" / "images"
    test_images = base_path / "test" / "images"
    if train_images.exists() and (val_images.exists() or test_images.exists()):
        print("Dataset already appears to be in YOLO format. Skipping formatting.")
        return

    candidates = discover_source_dirs(base_path)
    if not candidates:
        raise RuntimeError(
            "No candidate directories with paired images and labels were found. "
            "Ensure the dataset contains both image files and YOLO-format label files."
        )

    source_dir = max(candidates, key=lambda p: sum(1 for _ in p.glob("*")))
    print(f"Using source directory: {source_dir}")

    images = sorted(
        p for p in source_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )
    if not images:
        raise RuntimeError("No image files detected in the source directory.")

    random.seed(seed)
    random.shuffle(images)

    split_idx = int(len(images) * train_split)
    train_items = images[:split_idx] or images
    val_items = images[split_idx:] or images[: len(images) // 5] or images

    ensure_clean_targets(base_path)
    copy_split(train_items, base_path, "train")
    copy_split(val_items, base_path, "val")

    print("Dataset formatting complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reformat datasets into YOLO directory structure")
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Path to the dataset directory that needs formatting.",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Proportion of images to use for the training split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    auto_format_dataset(args.base_dir, args.train_split, args.seed)


if __name__ == "__main__":
    main()
