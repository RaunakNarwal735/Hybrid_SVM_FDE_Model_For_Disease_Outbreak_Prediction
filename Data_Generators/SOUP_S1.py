import os
import json
import argparse
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from joblib import dump
import glob
import pandas as pd

# Assume these functions are defined elsewhere or will be added
# from Data_Generators.SOUP_S1_utils import build_training_matrix, prepare_xy, train_svr, compute_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Robust Time Series Cross-Validation for SOUP_S1")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--out_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument("--lags", type=int, default=10, help="Number of lags for features")
    parser.add_argument("--no_grid", action="store_true", help="Skip grid search for SVR")
    return parser.parse_args()

def build_training_matrix(data_dir, n_lags=7, target_override=None):
    rows = []
    sir_param_dict = {}
    superset_cols = set()
    files = glob.glob(os.path.join(data_dir, "**", "dataset.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No dataset.csv files found in {data_dir} or subfolders.")
    # Add tqdm progress bar here
    for fpath in tqdm(sorted(files), desc="Processing datasets"):
        df = pd.read_csv(fpath).reset_index(drop=True)
        target_col = target_override or detect_target_col(df)
        y = df[target_col].values.astype(float)
        try:
            N = infer_population(df)
        except Exception:
            N = max(y) * 50 + 1.0
        beta, gamma, sir_pred = fit_sir_to_series(y, N=N)
        sir_param_dict[fpath] = {"beta": float(beta), "gamma": float(gamma), "N": float(N)}
        residual = y - sir_pred
        df2 = df.copy()
        df2["Target"] = y
        df2["SIR_Base"] = sir_pred
        df2["Residual"] = residual
        # Add lags for Target, SIR_Base, Infected (if present)
        for lag in range(1, n_lags + 1):
            df2[f"Target_lag{lag}"] = df2["Target"].shift(lag)
            df2[f"SIR_Base_lag{lag}"] = df2["SIR_Base"].shift(lag)
            if "Infected" in df2.columns:
                df2[f"Infected_lag{lag}"] = df2["Infected"].shift(lag)
        # Moving averages and std
        df2["MA7_Target"] = df2["Target"].rolling(7, min_periods=1).mean().shift(1)
        df2["STD7_Target"] = df2["Target"].rolling(7, min_periods=1).std().shift(1)
        # Calendar features
        df2["DayOfWeek"] = df2["Day"] % 7 if "Day" in df2.columns else np.arange(len(df2)) % 7
        # Tag dataset ID as folder name (e.g., run_001)
        df2["Dataset_ID"] = os.path.basename(os.path.dirname(fpath))
        rows.append(df2)
        superset_cols.update(df2.columns.tolist())
    big_df = pd.concat(rows, ignore_index=True)
    preferred_order = ["Dataset_ID", "Day", "Target", "SIR_Base", "Residual"]
    other_cols = [c for c in big_df.columns if c not in preferred_order]
    big_df = big_df[preferred_order + other_cols]
    return big_df, sir_param_dict, target_col

def prepare_xy(train_df, test_df, n_lags):
    # This function is assumed to be defined elsewhere
    # It should return X_train, y_train, X_test, y_test, feature_cols
    pass

def train_svr(X_train, y_train, grid_search=True):
    # This function is assumed to be defined elsewhere
    # It should return model, params
    pass

def compute_metrics(y_true, y_base, y_pred):
    # This function is assumed to be defined elsewhere
    # It should return a dictionary of metrics
    pass

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build training matrix
    big_df, sir_params, target_col = build_training_matrix(args.data_dir, n_lags=args.lags)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    metrics_list = []
    all_feature_cols = None
    for fold, (train_idx, test_idx) in enumerate(tqdm(tscv.split(big_df), total=tscv.get_n_splits(), desc='Training Progress')):
        train_df = big_df.iloc[train_idx]
        test_df = big_df.iloc[test_idx]
        X_train, y_train, X_test, y_test, feature_cols = prepare_xy(train_df, test_df, n_lags=args.lags)
        if all_feature_cols is None:
            all_feature_cols = feature_cols
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model, params = train_svr(X_train_s, y_train, grid_search=not args.no_grid)
        svr_resid_pred = model.predict(X_test_s)
        sir_base_test = test_df["SIR_Base"].values
        hybrid_pred = sir_base_test + svr_resid_pred
        hybrid_pred = np.clip(hybrid_pred, 0, None)
        y_true_test = test_df["Target"].values
        metrics = compute_metrics(y_true_test, sir_base_test, hybrid_pred)
        metrics_list.append(metrics)
    # Aggregate metrics
    def agg_metric(metric_name, model_name):
        vals = [m[model_name][metric_name] for m in metrics_list]
        return np.mean(vals), np.std(vals)
    print("\n=== Cross-Validation Results ===")
    for model in ["SIR", "Hybrid"]:
        for metric in ["RMSE", "MAE", "R2"]:
            mean, std = agg_metric(metric, model)
            print(f"{model} {metric}: {mean:.4f} ± {std:.4f}")
    # Save last model/scaler/params for deployment
    dump(model, os.path.join(args.out_dir, "hybrid_svr.joblib"))
    dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    with open(os.path.join(args.out_dir, "sir_params.json"), "w") as f:
        json.dump(sir_params, f, indent=4)
    with open(os.path.join(args.out_dir, "training_metadata.json"), "w") as f:
        json.dump({
            "target_used": target_col,
            "lags": args.lags,
            "svr_params": params,
            "feature_cols": all_feature_cols,
            "cv_metrics": metrics_list
        }, f, indent=4)
    print(f"Artifacts saved in: {args.out_dir}")

if __name__ == "__main__":
    main() 