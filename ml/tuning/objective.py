import time
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgbm
from sklearn.metrics import mean_squared_error

from processing.validator import TARGET_COL, get_feature_cols

def _rmsle_from_log_target(y_log_true, y_log_pred):
    # Since we train on log1p(units), RMSE in log space is equivalent to RMSLE in original units.
    return float(np.sqrt(mean_squared_error(y_log_true, y_log_pred)))

def make_objective(train_path, valid_path):
    # Load data once (caching) to speed up iterations
    print(f"Loading data for tuning from {train_path} and {valid_path}...")
    df_train = pd.read_parquet(train_path)
    df_valid = pd.read_parquet(valid_path)

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

    def objective(trial: optuna.Trial):
        start_time = time.time()

        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": 2025,
            
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 5e-2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "n_estimators": 50, # Set to a reasonable number for tuning
        }

        callbacks = []
        n_est = int(params["n_estimators"])
        if n_est > 5:
            callbacks.append(lgbm.early_stopping(stopping_rounds=50))
            
        model = lgbm.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            categorical_feature=["store_id", "item_id"],
            callbacks=callbacks
        )

        preds_log = model.predict(X_valid)
        score = _rmsle_from_log_target(y_valid, preds_log)

        duration = time.time() - start_time
        print(f"Trial {trial.number} finished with RMSLE: {score:.4f} (Duration: {duration:.2f}s)")

        return score

    return objective
