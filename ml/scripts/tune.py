import argparse
import json
import os
import time
import sys

import optuna

# Add current directory (ml) to path for imports
sys.path.append(os.getcwd())

from tuning.objective import make_objective


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-name", default="lgbm_global_optuna", help="Optuna study name")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials for Optuna study")
    parser.add_argument("--timeout-sec", type=int, default=0, help="Timeout in seconds for Optuna study (0 for no timeout)")
    parser.add_argument("--train-path", default="../shared/data/processed/train.parquet", help="Path to training data")
    parser.add_argument("--valid-path", default="../shared/data/processed/valid.parquet", help="Path to validation data")
    parser.add_argument("--out-best-params", default="../shared/outputs/tuning/best_params.json", help="Path to save best parameters")
    return parser.parse_args()


def champion_callback(study, trial):
    """Callback to print when a new champion (best trial) is found."""
    if study.best_trial.number == trial.number:
        print(f"🏆 [CHAMPION] trial={trial.number} value={trial.value:.4f}")


def tune_logic(args):
    """Core tuning logic using Optuna."""
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

    # Save best parameters
    os.makedirs(os.path.dirname(args.out_best_params), exist_ok=True)
    with open(args.out_best_params, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    
    print("-" * 30)
    print(f"Optimization finished.")
    print(f"Best Score (RMSLE): {study.best_value:.4f}")
    print(f"Best Params saved to: {args.out_best_params}")


def main():
    args = parse_args()
    tune_logic(args)


if __name__ == "__main__":
    main()
