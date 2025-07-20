#!/usr/bin/env python


import os
import json
import argparse
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from joblib import dump

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ------------------------------------------------------------------
# SEIR Utilities (replaces SIR)
# ------------------------------------------------------------------
def _seir_forward(beta, sigma, gamma, N, E0, I0, R0, T):
    """Discrete-time SEIR forward simulation, returns arrays (S,E,I,R) length T."""
    S, E, I, R = N - E0 - I0 - R0, E0, I0, R0
    S_list, E_list, I_list, R_list = [S], [E], [I], [R]
    for _ in range(1, T):
        dS = -beta * S * I / N
        dE = beta * S * I / N - sigma * E
        dI = sigma * E - gamma * I
        dR = gamma * I
        S += dS
        E += dE
        I += dI
        R += dR
        S_list.append(S)
        E_list.append(E)
        I_list.append(I)
        R_list.append(R)
    return np.array(S_list), np.array(E_list), np.array(I_list), np.array(R_list)

def _seir_curve_for_fit(t, beta, sigma, gamma, N, E0, I0, R0):
    """Wrapper to return I(t) for curve_fit."""
    _, _, I, _ = _seir_forward(beta, sigma, gamma, N, E0, I0, R0, len(t))
    return I

def fit_seir_to_series(y, N):
    """
    Fit SEIR to observed series y (target). Returns beta, sigma, gamma, seir_I_pred.
    y should be non-negative.
    """
    T = len(y)
    t = np.arange(T)
    I0 = max(float(y[0]), 1.0)
    E0 = I0  # assume E0 = I0 for initialization
    R0 = 0.0
    def f(t, beta, sigma, gamma):
        return _seir_curve_for_fit(t, beta, sigma, gamma, N, E0, I0, R0)
    popt, _ = curve_fit(
        f,
        t,
        y,
        p0=[0.2, 0.2, 0.1],
        bounds=(0.0, [2.0, 1.0, 1.0]),
        maxfev=20_000,
    )
    beta_opt, sigma_opt, gamma_opt = popt
    seir_pred = f(t, beta_opt, sigma_opt, gamma_opt)
    return beta_opt, sigma_opt, gamma_opt, seir_pred


# ------------------------------------------------------------------
# Feature Engineering
# ------------------------------------------------------------------
def add_lags(df, col, n_lags):
    """Add col_lag1..n to df; returns modified copy."""
    out = df.copy()
    for lag in range(1, n_lags + 1):
        out[f"{col}_lag{lag}"] = out[col].shift(lag)
    return out


def infer_population(df):
    """
    Infer population N from S+I+R columns if available.
    Fallback: max of any present compartment.
    """
    cols = ["Susceptible", "Infected", "Recovered"]
    have = [c for c in cols if c in df.columns]
    if have:
        return float(df[have].iloc[0].sum())
    # fallback: try any one
    for c in have:
        return float(df[c].iloc[0])
    raise ValueError("Cannot infer population: no S/I/R columns present.")


def detect_target_col(df):
    """Pick primary target column: Reported_Cases preferred, else Infected, else raise."""
    if "Reported_Cases" in df.columns:
        return "Reported_Cases"
    if "Infected" in df.columns:
        return "Infected"
    raise ValueError("No Reported_Cases or Infected column found for target.")


def numeric_feature_candidates(df):
    """
    Return list of numeric feature candidate columns EXCLUDING the target.
    We'll later drop lags & target/residual.
    """
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# ------------------------------------------------------------------
# Batch Loader & Training Set Builder
# ------------------------------------------------------------------
import glob

def build_training_matrix(data_dir, n_lags=7, target_override=None):
    """
    Load all CSVs in data_dir (recursively), build concatenated training DataFrame.
    """
    rows = []
    sir_param_dict = {}
    superset_cols = set()

    # Recursively find dataset.csv files
    files = glob.glob(os.path.join(data_dir, "**", "dataset.csv"), recursive=True)
    if not files:
        raise FileNotFoundError(f"No dataset.csv files found in {data_dir} or subfolders.")

    for fpath in sorted(files):
        df = pd.read_csv(fpath).reset_index(drop=True)

        target_col = target_override or detect_target_col(df)
        y = df[target_col].values.astype(float)

        try:
            N = infer_population(df)
        except Exception:
            N = max(y) * 50 + 1.0

        # Fit SEIR instead of SIR
        beta, sigma, gamma, seir_pred = fit_seir_to_series(y, N=N)
        sir_param_dict[fpath] = {"beta": float(beta), "sigma": float(sigma), "gamma": float(gamma), "N": float(N)}
        residual = y - seir_pred

        df2 = df.copy()
        df2["Target"] = y
        df2["SEIR_Base"] = seir_pred
        df2["Residual"] = residual

        # Add lags for Target, SEIR_Base, Exposed (if present)
        for lag in range(1, n_lags + 1):
            df2[f"Target_lag{lag}"] = df2["Target"].shift(lag)
            df2[f"SEIR_Base_lag{lag}"] = df2["SEIR_Base"].shift(lag)
            if "Exposed" in df2.columns:
                df2[f"Exposed_lag{lag}"] = df2["Exposed"].shift(lag)

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
    preferred_order = ["Dataset_ID", "Day", "Target", "SEIR_Base", "Residual"]
    other_cols = [c for c in big_df.columns if c not in preferred_order]
    big_df = big_df[preferred_order + other_cols]

    return big_df, sir_param_dict, target_col


# ------------------------------------------------------------------
# Train/Test Split (chrono, per dataset)
# ------------------------------------------------------------------
def split_train_test(big_df, test_frac=0.2):
    """
    Chronological split within each dataset; returns train_df, test_df.
    """
    train_parts = []
    test_parts = []
    for ds, sdf in big_df.groupby("Dataset_ID"):
        n = len(sdf)
        n_test = max(1, int(np.floor(n * test_frac)))
        n_train = n - n_test
        train_parts.append(sdf.iloc[:n_train].copy())
        test_parts.append(sdf.iloc[n_train:].copy())
    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


# ------------------------------------------------------------------
# Prepare X, y matrices
# ------------------------------------------------------------------
def prepare_xy(train_df, test_df, n_lags):
    """
    Build consistent feature list across train/test.
    We include:
        - SEIR_Base
        - All available context numeric columns EXCEPT:
              Dataset_ID, Day, Target, Residual
        - Lagged Target columns (Target_lag1..)
    Missing columns in either set are added & filled with 0.
    """
    drop_cols = {"Dataset_ID", "Target", "Residual"}
    # we keep Day (time index) as optional feature? Up to you; let's include scaled Day.
    # We'll include Day; it's numeric; remove if undesired.

    # Identify lag columns
    lag_cols = [c for c in train_df.columns if c.startswith("Target_lag") or c.startswith("SEIR_Base_lag") or c.startswith("Exposed_lag")]

    # numeric columns
    train_num = [c for c in train_df.columns if pd.api.types.is_numeric_dtype(train_df[c])]
    # candidate features = numeric minus drop_cols plus lag cols explicitly
    feature_cols = []
    for c in train_num:
        if c in drop_cols:
            continue
        feature_cols.append(c)

    # Add context columns if present
    for col in ["Susceptible", "Exposed", "Infected", "Recovered", "Beta_Effective", "Season_Index", "Intervention_Flag"]:
        if col in train_df.columns and col not in feature_cols:
            feature_cols.append(col)

    # guarantee lag order
    feature_cols = [c for c in feature_cols if c not in lag_cols] + sorted(lag_cols, key=lambda x: (x.split('_')[0], int(x.split('lag')[-1])) if 'lag' in x else (x, 0))

    # align test
    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = np.nan
    for c in feature_cols:
        if c not in train_df.columns:
            train_df[c] = np.nan

    # fill missing with 0
    train_mat = train_df[feature_cols].fillna(0.0).values
    test_mat = test_df[feature_cols].fillna(0.0).values

    # targets = residuals
    y_train = train_df["Residual"].values
    y_test = test_df["Residual"].values

    return train_mat, y_train, test_mat, y_test, feature_cols


# ------------------------------------------------------------------
# Train SVR
# ------------------------------------------------------------------
def train_svr(X_train, y_train, grid_search=True):
    if grid_search:
        param_grid = {
            "C": [1, 10, 100, 1000, 10000],
            "epsilon": [0.01, 0.05, 0.1, 0.5, 1.0],
            "gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
            "kernel": ["rbf", "linear", "poly"],
        }
        svr = SVR()
        grid = GridSearchCV(
            svr,
            param_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        params = grid.best_params_
    else:
        model = SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")
        model.fit(X_train, y_train)
        params = {"C": 100, "epsilon": 0.1, "gamma": "scale", "kernel": "rbf"}
    return model, params


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(y_true, sir_base, hybrid_pred):
    """Return dict of metrics for SIR-only and Hybrid."""
    sir_mse = mean_squared_error(y_true, sir_base)
    sir_mae = mean_absolute_error(y_true, sir_base)
    sir_r2 = r2_score(y_true, sir_base)

    hyb_mse = mean_squared_error(y_true, hybrid_pred)
    hyb_mae = mean_absolute_error(y_true, hybrid_pred)
    hyb_r2 = r2_score(y_true, hybrid_pred)

    return {
        "SIR": {"RMSE": float(np.sqrt(sir_mse)), "MAE": float(sir_mae), "R2": float(sir_r2)},
        "Hybrid": {"RMSE": float(np.sqrt(hyb_mse)), "MAE": float(hyb_mae), "R2": float(hyb_r2)},
    }


def per_dataset_metrics(test_df, hybrid_pred):
    """
    Compute per-dataset metrics vs SIR_Base and Hybrid.
    hybrid_pred is aligned to test_df rows order.
    """
    out = {}
    y_true_all = test_df["Target"].values
    sir_base_all = test_df["SEIR_Base"].values

    # Add hybrid to frame to group
    tmp = test_df.copy()
    tmp["_hybrid"] = hybrid_pred

    for ds, sdf in tmp.groupby("Dataset_ID"):
        yt = sdf["Target"].values
        sb = sdf["SEIR_Base"].values
        hp = sdf["_hybrid"].values
        out[ds] = compute_metrics(yt, sb, hp)
    return out


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Batch-train Hybrid SIR+SVR residual model.")
    ap.add_argument("--data_dir", required=True, help="Directory containing epidemic CSV files.")
    ap.add_argument("--out_dir", required=True, help="Where to save trained model & logs.")
    ap.add_argument("--lags", type=int, default=7, help="Number of lag days for target.")
    ap.add_argument("--test_frac", type=float, default=0.2, help="Fraction per dataset reserved for test.")
    ap.add_argument("--no_grid", action="store_true", help="Disable GridSearch; use fixed SVR params.")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Build training matrix
    big_df, sir_params, target_col = build_training_matrix(args.data_dir, n_lags=args.lags)

    # Split chronologically within each dataset
    train_df, test_df = split_train_test(big_df, test_frac=args.test_frac)

    # Prepare matrices
    X_train, y_train, X_test, y_test, feature_cols = prepare_xy(train_df, test_df, n_lags=args.lags)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train SVR on residuals
    model, params = train_svr(X_train_s, y_train, grid_search=not args.no_grid)

    # Predict residual corrections
    svr_resid_pred = model.predict(X_test_s)

    # Build hybrid predictions: SEIR_Base + SVR residual
    seir_base_test = test_df["SEIR_Base"].values
    hybrid_pred = seir_base_test + svr_resid_pred
    hybrid_pred = np.clip(hybrid_pred, 0, None)

    # Evaluate
    y_true_test = test_df["Target"].values
    overall_metrics = compute_metrics(y_true_test, seir_base_test, hybrid_pred)
    ds_metrics = per_dataset_metrics(test_df, hybrid_pred)

    # Save artifacts
    dump(model, os.path.join(args.out_dir, "hybrid_svr.joblib"))
    dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))

    with open(os.path.join(args.out_dir, "sir_params.json"), "w") as f:
        json.dump(sir_params, f, indent=4)

    # Predictions CSV (test rows only)
    pred_df = test_df.copy()
    pred_df["Hybrid_Pred"] = hybrid_pred
    pred_df["Hybrid_Residual_Pred"] = svr_resid_pred
    pred_df.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)

    # Metrics JSON
    meta = {
        "target_used": target_col,
        "lags": args.lags,
        "test_frac": args.test_frac,
        "svr_params": params,
        "feature_cols": feature_cols,
        "overall_metrics": overall_metrics,
        "per_dataset_metrics": ds_metrics,
        "n_train_rows": int(len(y_train)),
        "n_test_rows": int(len(y_test)),
        "n_datasets": int(len(sir_params)),
    }
    with open(os.path.join(args.out_dir, "training_metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("\n=== Hybrid Batch Training Complete ===")
    print(f"Datasets used: {meta['n_datasets']}")
    print(f"Train rows: {meta['n_train_rows']}  Test rows: {meta['n_test_rows']}")
    print(f"Target: {target_col}")
    print("SVR params:", params)
    print("Overall Metrics:", overall_metrics)
    print(f"Artifacts saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
