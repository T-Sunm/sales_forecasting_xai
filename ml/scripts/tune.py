import argparse
import json
import os
import time
import sys

import mlflow
import optuna

# Add current directory (ml) to path for imports
sys.path.append(os.getcwd())

from utils.mlflow_utils import setup_mlflow
from tuning.objective import make_objective

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", default="walmart-sales-tuning", help="MLflow experiment name")
    parser.add_argument("--study-name", default="lgbm_global_optuna", help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna study")
    parser.add_argument("--timeout-sec", type=int, default=0, help="Timeout in seconds for Optuna study (0 for no timeout)")
    parser.add_argument("--train-path", default="../data/processed/train.parquet", help="Path to training data")
    parser.add_argument("--valid-path", default="../data/processed/valid.parquet", help="Path to validation data")
    parser.add_argument("--out-best-params", default="outputs/tuning/best_params.json", help="Path to save best parameters")
    return parser.parse_args()

def champion_callback(study, trial):
    """Callback to print when a new champion (best trial) is found."""
    if study.best_trial.number == trial.number:
        print(f"🏆 [CHAMPION] trial={trial.number} value={trial.value:.4f}")

def tune_logic(args):
    """Core tuning logic to be executed within an MLflow run context."""
    # Set business-level tags for easy filtering
    mlflow.set_tag("project", "walmart-sales-forecasting")
    mlflow.set_tag("stage", "tune")
    mlflow.set_tag("model_family", "lightgbm")
    mlflow.set_tag("study_name", args.study_name)
    mlflow.set_tag("target_space", "log1p_units")

    # Load data and prepare objective
    objective = make_objective(args.train_path, args.valid_path)

    # Create and run Optuna study
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
    )
    
    timeout = args.timeout_sec if args.timeout_sec > 0 else None
    
    print(f"Starting tuning study '{args.study_name}' with {args.n_trials} trials...")
    study.optimize(
        objective, 
        n_trials=args.n_trials, 
        timeout=timeout, 
        callbacks=[champion_callback]
    )

    # Log best results to parent run
    mlflow.log_metric("best_val_rmsle", study.best_value)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    
    # Log trial pointers for easy traceability
    mlflow.set_tag("best_trial_number", study.best_trial.number)
    best_run_id = study.best_trial.user_attrs.get("mlflow_run_id")
    if best_run_id:
        mlflow.set_tag("best_child_run_id", best_run_id)

    # Save and log best parameters as artifact
    os.makedirs(os.path.dirname(args.out_best_params), exist_ok=True)
    with open(args.out_best_params, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    
    mlflow.log_artifact(args.out_best_params)

    print("-" * 30)
    print(f"Optimization finished.")
    print(f"Best Score (RMSLE): {study.best_value:.4f}")
    print(f"Best Params saved to: {args.out_best_params}")

def main():
    args = parse_args()
    setup_mlflow()
    
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    
    # When running via 'mlflow run', an active run is already started by the CLI.
    # We use that as the parent. If running directly, we start a new one.
    run = mlflow.active_run()
    if run:
        print(f"Using active MLflow run (managed by CLI): {run.info.run_id}")
        tune_logic(args)
    else:
        print("No active run detected. Starting a new session...")
        mlflow.set_experiment(args.experiment_name)
        parent_run_name = f"optuna_study_{args.study_name}_{int(time.time())}"
        with mlflow.start_run(run_name=parent_run_name):
            tune_logic(args)

if __name__ == "__main__":
    main()
