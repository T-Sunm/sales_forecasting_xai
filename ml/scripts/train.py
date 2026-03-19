import os
import sys
import json
import time
import argparse

import pandas as pd
import numpy as np
import yaml
import joblib
import lightgbm as lgbm
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(os.getcwd())

from processing.validator import TARGET_COL, get_feature_cols

PARAMS_FILE = "../shared/params.yaml"


def load_train_params(best_params_path=None):
    if not os.path.exists(PARAMS_FILE):
        return {}
    with open(PARAMS_FILE, "r") as f:
        all_params = yaml.safe_load(f)
    cfg = all_params.get("train", {})
    
    if best_params_path and os.path.exists(best_params_path):
        with open(best_params_path, "r") as f:
            best_params = json.load(f)
        cfg.update(best_params)
        print(f"Loaded best_params from {best_params_path}")
    
    return cfg


def train_logic(args):
    start_time = time.time()

    if not os.path.exists(args.train_path):
        print(f"Error: {args.train_path} not found. Run prepare_data.py first.")
        return None, None

    print(f"Loading data from {args.train_path} and {args.valid_path}...")
    df_train = pd.read_parquet(args.train_path)
    df_valid = pd.read_parquet(args.valid_path)

    feature_cols = get_feature_cols(df_train.columns)
    extra_features = ["store_id", "item_id"]

    X_train = df_train[feature_cols + extra_features].copy()
    y_train = df_train[TARGET_COL]
    X_valid = df_valid[feature_cols + extra_features].copy()
    y_valid = df_valid[TARGET_COL]

    # Ensure categorical types for LightGBM
    for c in ["store_id", "item_id"]:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")

    cfg = load_train_params(args.best_params)
    params = {
        "objective": cfg.get("objective", "regression"),
        "metric": cfg.get("metric", "rmse"),
        "boosting_type": cfg.get("boosting_type", "gbdt"),
        "verbosity": -1,
        "num_leaves": cfg.get("num_leaves", 127),
        "learning_rate": cfg.get("learning_rate", 0.0125),
        "feature_fraction": cfg.get("feature_fraction", 0.8),
        "bagging_fraction": cfg.get("bagging_fraction", 0.9),
        "bagging_freq": cfg.get("bagging_freq", 5),
        "min_child_samples": cfg.get("min_child_samples", 20),
        "lambda_l1": cfg.get("lambda_l1", 0.1),
        "lambda_l2": cfg.get("lambda_l2", 0.1),
        "max_depth": cfg.get("max_depth", -1),
        "n_estimators": cfg.get("n_estimators", 100),
        "random_state": cfg.get("random_state", 2025),
    }
    early_stopping_rounds = cfg.get("early_stopping_rounds", 50)

    print("Training model...")
    model = lgbm.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgbm.early_stopping(stopping_rounds=early_stopping_rounds),
            lgbm.log_evaluation(period=50),
        ],
        categorical_feature=["store_id", "item_id"],
    )

    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    rmsle = np.sqrt(mean_squared_error(y_valid, preds))
    train_time = time.time() - start_time

    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSLE: {rmsle:.4f}")

    metrics = {
        "val_mae": round(mae, 6),
        "val_rmsle": round(rmsle, 6),
        "train_rows": len(df_train),
        "valid_rows": len(df_valid),
        "n_features": len(feature_cols) + len(extra_features),
        "train_time_sec": float(f"{train_time:.2f}"),
        "best_iteration": model.best_iteration_ if hasattr(model, "best_iteration_") else params["n_estimators"],
    }

    return model, metrics


def save_artifacts(model, metrics, model_out, metrics_out):
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(model, model_out)
    print(f"Model saved to {model_out}")

    os.makedirs(os.path.dirname(metrics_out), exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="../shared/data/processed/train.parquet")
    parser.add_argument("--valid-path", default="../shared/data/processed/valid.parquet")
    parser.add_argument("--model-out", default="../shared/models/lgbm_baseline.pkl")
    parser.add_argument("--metrics-out", default="../shared/outputs/metrics.json")
    parser.add_argument("--best-params", default=None)
    
    args = parser.parse_args()

    model, metrics = train_logic(args)

    if model and metrics:
        save_artifacts(model, metrics, args.model_out, args.metrics_out)
