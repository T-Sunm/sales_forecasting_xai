import os
import pandas as pd
import numpy as np
import lightgbm as lgbm
import mlflow
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error

import argparse

# Add current directory (ml) to path for imports
sys.path.append(os.getcwd())

from utils.mlflow_utils import setup_mlflow
from processing.validator import TARGET_COL, get_feature_cols

def train(experiment_name, train_path, valid_path, run_name):
    setup_mlflow()
    mlflow.set_experiment(experiment_name)
    mlflow.lightgbm.autolog()
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Run prepare_data.py first.")
        return

    print(f"Loading prepared data from {train_path} and {valid_path}...")
    df_train = pd.read_parquet(train_path)
    df_valid = pd.read_parquet(valid_path)
    
    feature_cols = get_feature_cols(df_train.columns)
    extra_features = ['store_id', 'item_id']
    
    X_train = df_train[feature_cols + extra_features]
    y_train = df_train[TARGET_COL]
    
    X_valid = df_valid[feature_cols + extra_features]
    y_valid = df_valid[TARGET_COL]
    
    # Define baseline params (from notebook's optuna result)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": 127,
        "learning_rate": 0.012528590527215604,
        "feature_fraction": 0.8044026085648729,
        "bagging_fraction": 0.9028865865109351,
        "bagging_freq": 7,
        "min_child_samples": 42,
        "lambda_l1": 7.327783159306205e-08,
        "lambda_l2": 0.0035687388325977474,
        "max_depth": 11,
        "n_estimators": 2000,
        "random_state": 2025
    }
    
    print(f"Starting training baseline model in experiment '{experiment_name}' with run name '{run_name}'...")
    with mlflow.start_run(run_name=run_name):
        model = lgbm.LGBMRegressor(**params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                lgbm.early_stopping(stopping_rounds=50),
                lgbm.log_evaluation(period=50)
            ],
            categorical_feature=['store_id', 'item_id']
        )
        
        preds = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, preds)
        # RMSE on log-space is mathematically equivalent to RMSLE on original units
        rmsle = np.sqrt(mean_squared_error(y_valid, preds))
        
        print(f"Validation MAE: {mae:.4f}")
        print(f"Validation RMSLE (RMSE on log-target): {rmsle:.4f}")
        
        mlflow.log_metric("val_mae", mae)
        mlflow.log_metric("val_rmsle", rmsle)
        
        print("Training complete and logged to MLflow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="walmart-sales-baseline", help="MLflow experiment name")
    parser.add_argument("--train-path", default="../data/processed/train.parquet", help="Path to training parquet file")
    parser.add_argument("--valid-path", default="../data/processed/valid.parquet", help="Path to validation parquet file")
    parser.add_argument("--run-name", default="lgbm_baseline_global", help="MLflow run name")
    args = parser.parse_args()
    
    train(
        experiment_name=args.experiment_name,
        train_path=args.train_path,
        valid_path=args.valid_path,
        run_name=args.run_name
    )
