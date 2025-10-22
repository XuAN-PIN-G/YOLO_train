# YOLO_train

YOLO_train provides a reusable YOLOv8 training scaffold. It keeps the original automation flow intact while stripping away domain-specific assumptions. Any developer can copy a single configuration file, adjust dataset metadata and hyperparameters, and launch training without touching the code.

## Key Features

- **Configuration-driven**: define dataset locations, class names, and training hyperparameters through one YAML file.
- **Automated data pipeline**: optionally downloads Kaggle datasets, reshapes directory layouts, and generates the Ultralytics `data.yaml`.
- **Plug-and-play**: supports every YOLOv8 checkpoint (`yolov8n.pt` → `yolov8x.pt`) across CPU, CUDA, and MPS backends.
- **Extensible design**: modular scripts make it easy to integrate into CI/CD or extend for custom workflows.

## Project Layout

```
YOLO_train/
├── configs/                # Example and user-defined configurations
│   └── sample.yaml
├── data/                   # Default dataset directory (overridable via config)
├── runs/                   # YOLO training outputs
├── scripts/
│   ├── auto_format_dataset.py
│   ├── download_dataset.py
│   ├── prepare_data.py
│   └── train.py
├── requirements.txt
└── README.md
```

## Quick Start

1. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Copy the configuration template**

   ```bash
   cp configs/sample.yaml configs/my_dataset.yaml
   ```

   Adjust the `dataset` and `training` sections to fit your dataset (details below).

3. **Download or prepare the dataset (optional)**

   For Kaggle-hosted datasets that are not yet downloaded:

   ```bash
   python scripts/download_dataset.py --config configs/my_dataset.yaml
   ```

   If the dataset already exists locally, ensure it follows the Ultralytics layout (see next section), or run:

   ```bash
   python scripts/prepare_data.py --config configs/my_dataset.yaml
   ```

4. **Start training**

   ```bash
   python scripts/train.py --config configs/my_dataset.yaml
   ```

   Training logs and checkpoints will appear in the `runs/` directory.

## Configuration

`configs/sample.yaml` provides a minimal example. Field descriptions:

```yaml
dataset:
  name: custom-project          # Human-readable dataset identifier
  provider: kaggle              # Data source: kaggle or local
  slug: owner/dataset-name      # Kaggle dataset slug (required when provider=kaggle)
  local_dir: data/custom        # Local dataset path
  classes:                      # Class names in index order
    - class-a
    - class-b

training:
  model: yolov8n.pt             # Pretrained checkpoint
  imgsz: 640                    # Square input resolution
  batch: 16                     # Batch size
  epochs: 100                   # Training epochs
  device: cuda                  # Training device: cpu, cuda, mps, etc.
```

> **Tip**: to expose additional YOLO arguments, fork the project and extend `scripts/train.py` so they map to configuration keys.

## Dataset Preparation Guide

YOLOv8 expects the dataset layout below:

```
<local_dir>/
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

If a Kaggle dataset does not ship in that structure, run:

```bash
python scripts/auto_format_dataset.py --base-dir data/custom
```

The helper scans for mixed image/label folders, performs an 80/20 train-val split, and copies paired files into the YOLO format.

## Kaggle Credentials

The project can load Kaggle credentials from a `.env` file placed in the project root:

```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

The download script reads this file by default (or accepts `--env-file` for custom locations). You can also rely on the official Kaggle workflow by exporting environment variables or adding `~/.kaggle/kaggle.json`.

## FAQ

- **`Ultralytics package not found`**: confirm that `pip install -r requirements.txt` has been executed.
- **`Failed to authenticate with Kaggle API`**: verify environment variables or `.env`, or test authentication with the `kaggle` CLI.
- **`data.yaml` missing**: check the dataset layout and ensure `train/images` and `val/images` (or `test/images`) exist.

## License

When using this project publicly, follow the terms outlined in [LICENSE](../LICENSE). Contributions via issues or pull requests are welcome. Happy training!
