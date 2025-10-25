# Repository Guidelines

## Project Structure & Module Organization
`configs/` holds dataset + hyperparameter YAML templates; copy `sample.yaml` to create new runs. `scripts/` contains the automation pipeline (download, auto-format, data prep, train, and `run_pipeline.py`). Datasets live in `data/` (configurable) and raw YOLO outputs land in `runs/`. `requirements.txt` lists Ultralytics and helper libraries; keep it in sync with any code changes. Treat `README.md` as the single source for onboarding details—update it whenever paths or CLI flags change.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: standard environment setup before running anything else.
- `pip install -r requirements.txt`: installs the YOLO/utility stack; rerun after changing dependencies.
- `python scripts/run_pipeline.py --config configs/<name>.yaml`: executes download → format → prepare → train; supports `--env-file` for Kaggle creds and `--skip-*` flags.
- `python scripts/train.py --config ...`: runs YOLO training only; assumes `data.yaml` already exists.
- `python scripts/prepare_data.py --config ...`: regenerates `data.yaml` when labels or splits change.

## Coding Style & Naming Conventions
Follow PEP 8: 4-space indentation, snake_case for functions/variables, and UpperCamelCase only for classes. Keep scripts focused—one entrypoint per file with helper functions above `if __name__ == "__main__":`. Name new configs `configs/<dataset>.yaml` and prefer descriptive flag names (`--skip-download`, not abbreviations). Run `black` or `ruff format` before committing if you have them locally; otherwise ensure consistent spacing and docstrings for CLI arguments.

## Testing Guidelines
No dedicated unit test suite exists yet, so lean on reproducible smoke runs. Use `python scripts/run_pipeline.py --skip-train` to validate data ingestion quickly, then enable training for a short epoch count (e.g., set `training.epochs: 1`) to confirm end-to-end behavior. When altering dataset logic, create a throwaway config pointing to `data/sample` and document observed outputs in the PR. Aim for deterministic data splits; avoid random seeds unless they are explicitly set in the config.

## Commit & Pull Request Guidelines
Match the conventional commits style visible in history (`feat: add pipeline`, `docs: update readme`, etc.). Each commit should cover one logical change and include config/script updates plus doc edits together. Pull requests must describe: purpose, affected scripts/configs, manual verification steps, and any new CLI flags. Link issues when available and add screenshots or log excerpts if changes impact run output. Never commit `.env`, Kaggle tokens, or large artifacts—use `.gitignore` updates when introducing new generated directories.
