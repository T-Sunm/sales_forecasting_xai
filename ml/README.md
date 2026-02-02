# Machine Learning Core (MLflow Project)

This directory contains the core machine learning logic for sales forecasting, packaged as an MLflow Project.

## Structure

```text
ml/
├── MLproject                 # MLflow project definition
├── pyproject.toml            # ML-specific dependencies (uv)
├── uv.lock                   # uv lockfile
├── processing/               # Data validation & processing logic
│   ├── __init__.py
│   └── validator.py
├── scripts/                  # MLflow entry points scripts
│   ├── __init__.py
│   ├── prepare_data.py       # Data preparation script
│   └── train.py              # Model training script
└── utils/                    # Shared utilities
    ├── __init__.py
    └── mlflow_utils.py       # MLflow setup & config helpers
```

## Environment Management

This project uses **uv** for dependency management. To ensure your local environment matches the project requirements:

```bash
cd ml
uv sync
```

## Running the Project

### Using MLflow CLI (Recommended)

From the root of the repository, you can run the training entry point using MLflow:

```bash
mlflow run ./ml -e train --env-manager local
```

*Note: `--env-manager local` is used to leverage the environment already set up by uv.*

### Running Scripts Directly

You can also run the scripts as Python modules from the `ml/` directory:

```bash
cd ml
python -m scripts.train
```
