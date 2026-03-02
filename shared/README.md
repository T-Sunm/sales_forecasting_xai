# Shared — DVC Pipeline & Artifacts

> **Scope:** This directory is the **single source of truth for reproducibility**. It owns the DVC pipeline definition, all parameter values, raw input data, processed datasets, trained model artifacts, and evaluation metrics. No application code lives here — only pipeline orchestration, data, and artifacts.

---

## Overview

`shared/` is the central coordination point between the data preparation step (reading from PostgreSQL/Kaggle CSV), the ML model training pipeline (`ml/`), and the served artifacts consumed by `backend/`. DVC tracks every file's MD5 checksum and associates exact parameter values with each pipeline run, making every experiment fully reproducible from a single `dvc repro` command.

The DVC remote is **MinIO** (S3-compatible), storing cached artifacts at `s3://dvc-store/sales_forecasting_xai/shared`. Credentials for the remote are stored in `.dvc/config.local` (not committed to git).

---

## Directory Layout

```
shared/
├── dvc.yaml                    ← Pipeline definition (3 stages)
├── dvc.lock                    ← Locked checksums for all deps/outs/params
├── params.yaml                 ← All tunable parameters (prepare, train, tune)
├── .dvc/
│   ├── config                  ← Remote name + URL (committed)
│   └── config.local            ← Remote credentials (NOT committed — gitignored)
├── data/
│   ├── data_raw/
│   │   └── test.csv            ← Kaggle test IDs (dependency for prepare_data)
│   └── processed/              ← DVC-tracked outputs of prepare_data stage
│       ├── train.parquet       ← Training set (pre-cutoff rows, non-Kaggle)
│       ├── valid.parquet       ← Validation set (post-cutoff rows, non-Kaggle)
│       └── test.parquet        ← Kaggle test set (rows flagged is_kaggle_test=1)
├── models/
│   └── lgbm_baseline.pkl       ← DVC-tracked output of train stage (11.8 MB)
├── outputs/
│   ├── metrics.json            ← Training metrics (DVC metric, cache: false)
│   └── tuning/
│       └── best_params.json    ← Optuna best parameters (output of tune stage)
├── utils/
│   └── utils.py                ← Shared EDA utilities (rolling, EWMA, plotting)
├── notebooks/                  ← Exploratory notebooks (not tracked by DVC)
├── pyproject.toml              ← dvc>=3.66.1, dvc-s3>=3.3.0
└── .python-version             ← Python version pin
```

---

## DVC Pipeline Stages

Defined in `dvc.yaml`. Run all stages in order with:

```fish
cd shared
dvc repro
```

DVC automatically determines which stages are stale based on changed deps, params, or outs.

### Stage 1: `prepare_data`

| Field | Value |
|---|---|
| Command | `cd ../ml && uv run python -m scripts.prepare_data` |
| Dependencies | `../ml/scripts/prepare_data.py`, `data/data_raw/test.csv` |
| Parameters read | `prepare.cutoff_date` (from `params.yaml`) |
| Outputs | `data/processed/train.parquet`, `valid.parquet`, `test.parquet` |

**What it does:**
1. Reads feature matrix from PostgreSQL (`marts.sales_forecast JOIN marts.dim_date`).
2. Reads Kaggle test IDs from `data/data_raw/test.csv`.
3. Splits on `prepare.cutoff_date` (`2014-08-01`):
   - `train`: rows before cutoff, non-Kaggle
   - `valid`: rows after cutoff, non-Kaggle
   - `test`: rows flagged `is_kaggle_test=1`
4. Saves as Parquet to `data/processed/`.

> Source: `ml/scripts/prepare_data.py`. Connects to `POSTGRES_HOST:5432/sales_forecasting` using `.env` vars.

---

### Stage 2: `tune`

| Field | Value |
|---|---|
| Command | `cd ../ml && uv run python -m scripts.tune --train-path ../shared/data/processed/train.parquet --valid-path ../shared/data/processed/valid.parquet --out-best-params ../shared/outputs/tuning/best_params.json` |
| Dependencies | `../ml/scripts/tune.py`, `../ml/tuning/objective.py`, `data/processed/train.parquet`, `data/processed/valid.parquet` |
| Parameters read | `tune` block (from `params.yaml`) |
| Outputs | `outputs/tuning/best_params.json` |

**What it does:** Runs Optuna with `n_trials=50` to find optimal LightGBM hyperparameters. Each trial is logged as a nested MLflow run in study `lgbm_global_optuna`. Best params are serialised to `best_params.json`.

> **Note:** `tune` does **not** depend on `prepare_data` in the DAG graph order — both `tune` and `train` independently consume `data/processed/*.parquet`. Run order is `prepare_data → tune → train` because `train` depends on `outputs/tuning/best_params.json` (output of `tune`). Source: `dvc.yaml` lines 36–50.

---

### Stage 3: `train`

| Field | Value |
|---|---|
| Command | `cd ../ml && uv run python -m scripts.train --train-path ../shared/data/processed/train.parquet --valid-path ../shared/data/processed/valid.parquet --model-out ../shared/models/lgbm_baseline.pkl --metrics-out ../shared/outputs/metrics.json --best-params ../shared/outputs/tuning/best_params.json` |
| Dependencies | `../ml/scripts/train.py`, `../ml/processing/validator.py`, `data/processed/train.parquet`, `data/processed/valid.parquet`, `outputs/tuning/best_params.json` |
| Parameters read | `train` block (from `params.yaml`) |
| Outputs | `models/lgbm_baseline.pkl` |
| Metrics | `outputs/metrics.json` (`cache: false` — always re-evaluated) |

**What it does:** Trains a LightGBM regressor using the best Optuna params (merged with `params.yaml` defaults), logs autolog metrics to MLflow, and saves the pickle to `models/lgbm_baseline.pkl` (11.8 MB per `dvc.lock`).

---

## Locked State (`dvc.lock`)

The current committed lock reflects a successful full run:

| Artifact | MD5 | Size |
|---|---|---|
| `data/data_raw/test.csv` | `615f55aa` | 9.4 MB |
| `data/processed/train.parquet` | `40b6f2be` | 3.8 MB |
| `data/processed/valid.parquet` | `a90a8f87` | 279 KB |
| `data/processed/test.parquet` | `9b1428df` | 26 KB |
| `models/lgbm_baseline.pkl` | `4faa6533` | 11.8 MB |
| `outputs/metrics.json` | `245d9761` | 176 B |
| `outputs/tuning/best_params.json` | `a1178467` | 295 B |

> Source: `dvc.lock`.

---

## Parameters Reference (`params.yaml`)

### `prepare`

| Parameter | Current value | Effect |
|---|---|---|
| `cutoff_date` | `"2014-08-01"` | Train/valid split date. Change triggers re-run of `prepare_data` → `tune` → `train`. |

### `train`

| Parameter | Current value | Notes |
|---|---|---|
| `objective` | `regression` | LightGBM objective |
| `metric` | `rmse` | Evaluation metric |
| `boosting_type` | `gbdt` | Gradient boosting decision tree |
| `num_leaves` | `127` | Tree complexity |
| `learning_rate` | `0.0125` | Step size |
| `feature_fraction` | `0.804` | Column subsampling |
| `bagging_fraction` | `0.903` | Row subsampling |
| `bagging_freq` | `7` | Bagging frequency |
| `min_child_samples` | `42` | Leaf minimum samples |
| `lambda_l1` | `7.33e-08` | L1 regularization |
| `lambda_l2` | `0.00357` | L2 regularization |
| `max_depth` | `11` | Tree depth |
| `n_estimators` | `1000` | Max boosting rounds |
| `early_stopping_rounds` | `50` | Early stopping patience |
| `random_state` | `2025` | Reproducibility seed |

### `tune`

| Parameter | Current value | Notes |
|---|---|---|
| `n_trials` | `50` | Number of Optuna trials |
| `timeout_sec` | `0` | No timeout (0 = disabled) |
| `study_name` | `lgbm_global_optuna` | MLflow experiment + Optuna study name |

---

## DVC Remote Configuration

### Global config (`.dvc/config` — committed)

```ini
[core]
    remote = minio
['remote "minio"']
    url = s3://dvc-store/sales_forecasting_xai/shared
    endpointurl = http://localhost:9000
```

### Local credentials (`.dvc/config.local` — NOT committed, gitignored)

```ini
['remote "minio"']
    access_key_id = minioadmin
    secret_access_key = minioadmin
```

> **Important:** `.dvc/config.local` must exist on every developer's machine before `dvc push` / `dvc pull` will work. The bucket `dvc-store` must be created in MinIO before first push. **TODO:** Document the one-time bucket creation step (`mc mb minio/dvc-store` or MinIO Console at http://localhost:9001).

---

## Python Environment

| Dependency | Version | Purpose |
|---|---|---|
| `dvc` | `>=3.66.1` | Pipeline orchestration |
| `dvc-s3` | `>=3.3.0` | MinIO/S3 remote backend |

```fish
cd shared
uv sync
```

---

## How to Run (Local)

### Prerequisites

- MinIO running at `localhost:9000` with bucket `dvc-store` created.
- PostgreSQL running at `POSTGRES_HOST:5432` with `sales_forecasting` DB populated (mart tables from dbt).
- MLflow server running at `MLFLOW_TRACKING_URI` (for `tune` and `train` stages).
- `.env` at project root with `POSTGRES_*`, `MLFLOW_TRACKING_URI`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

### Full pipeline run

```fish
cd shared

# Run all stale stages in dependency order
dvc repro
```

### Run a single stage

```fish
# Only prepare_data
dvc repro prepare_data

# Only tune (requires train.parquet + valid.parquet)
dvc repro tune

# Only train (requires best_params.json + parquet files)
dvc repro train
```

### Force re-run a stage regardless of cache

```fish
dvc repro --force train
```

### Inspect current pipeline status

```fish
# See which stages are stale/up-to-date
dvc status

# Show stage DAG
dvc dag
```

### Push/Pull artifacts to MinIO remote

```fish
# Push all tracked files to MinIO
dvc push

# Pull artifacts from MinIO (restore data/models on a new machine)
dvc pull
```

### View tracked metrics

```fish
dvc metrics show
```

### Parameter experiment: change cutoff date

```fish
# Edit params.yaml: prepare.cutoff_date: "2013-08-01"
# Then repro — DVC auto-detects the change and re-runs all 3 stages
dvc repro
```

---

## Integration Points

| Component | Direction | Mechanism |
|---|---|---|
| `ml/scripts/prepare_data.py` | → reads from | PostgreSQL `marts.sales_forecast` + Kaggle CSV `data/data_raw/test.csv` |
| `ml/scripts/tune.py` | → reads from | `data/processed/train.parquet`, `data/processed/valid.parquet`; writes `outputs/tuning/best_params.json` |
| `ml/scripts/train.py` | → reads from | `data/processed/train.parquet`, `data/processed/valid.parquet`, `outputs/tuning/best_params.json`; writes `models/lgbm_baseline.pkl` |
| `backend/` | ← reads from | `models/lgbm_baseline.pkl` to serve predictions |
| MinIO DVC remote | push/pull | `s3://dvc-store/sales_forecasting_xai/shared` via `dvc-s3` |
| MLflow | → logs to | `tune` and `train` stages log metrics and artifacts to `MLFLOW_TRACKING_URI` |
| PostgreSQL (`prepare_data`) | ← reads from | `POSTGRES_HOST:5432/sales_forecasting` — mart tables populated by dbt warehouse project |

---

## Related README Files

| Link | Coverage |
|---|---|
| [../ml/README.md](../ml/README.md) | ML scripts invoked by DVC stages (train, tune, prepare_data) |
| [../data_platform/dbt/README.md](../data_platform/dbt/README.md) | dbt warehouse project that populates the DB read by prepare_data |
| [../data_platform/infra/postgres/README.md](../data_platform/infra/postgres/README.md) | PostgreSQL setup that stores dbt mart tables |
| [../data_platform/infra/spark_minio/README.md](../data_platform/infra/spark_minio/README.md) | MinIO that hosts the DVC remote (`dvc-store`) |
| [../backend/README.md](../backend/README.md) | FastAPI backend that loads `models/lgbm_baseline.pkl` |
| [../README.md](../README.md) | Root project overview and quickstart |
