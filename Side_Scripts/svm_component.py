import argparse
import os
import json

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from typing import Tuple, List


# ------------------------------------------------------------------
# Utility: parse CLI arguments
# ------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SVR for disease outbreak forecasting with Naive and optional SIR baselines."
    )
    parser.add_argument(
        "--data", required=True, help="Path to input CSV dataset."
    )
    parser.add_argument(
        "--logdir", default="logs", help="Directory to write predictions & metrics."
    )
    parser.add_argument(
        "--nlags", type=int, default=3, help="Number of lag features to create from target series."
    )
    parser.add_argument(
        "--train_frac", type=float, default=0.8, help="Fraction of rows (chronological) used for training."
    )
    parser.add_argument(
        "--no_grid", action="store_true",
        help="If passed, skip GridSearchCV and use default SVR(C=10,epsilon=0.1)."
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Step 0: Load data
# ------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Make sure Day exists; if not, create sequential index
    if "Day" not in df.columns:
        df.insert(0, "Day", np.arange(len(df)))
    return df


# ------------------------------------------------------------------
# Step 1: Determine modeling target
# ------------------------------------------------------------------
def detect_target(df: pd.DataFrame) -> str:
    if "Reported_Cases" in df.columns:
        return "Reported_Cases"
    elif "Infected" in df.columns:
        return "Infected"
    else:
        raise ValueError(
            "Dataset must contain either 'Reported_Cases' or 'Infected' column as modeling target."
        )


# ------------------------------------------------------------------
# Step 2: Create supervised dataset with lags + target (t+1)
# ------------------------------------------------------------------
def make_supervised(df: pd.DataFrame, target_col: str, n_lags: int = 3) -> pd.DataFrame:
    out = df.copy()

    # Lag features
    for lag in range(1, n_lags + 1):
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)

    # Target (next step)
    out["Target"] = out[target_col].shift(-1)

    # Drop rows with NaNs introduced by shifting
    out = out.dropna().reset_index(drop=True)
    return out


# ------------------------------------------------------------------
# Step 3: Select available feature columns safely
# ------------------------------------------------------------------
OPTIONAL_FEATURES = ["Beta_Effective", "Season_Index", "Intervention_Flag", "Reporting_Prob"]

def select_features(df: pd.DataFrame, target_col: str) -> List[str]:
    feature_cols = [c for c in df.columns if c.startswith(f"{target_col}_lag")]
    for col in OPTIONAL_FEATURES:
        if col in df.columns:
            feature_cols.append(col)
    # Ensure we don't accidentally include Target or Day
    drop_cols = {"Target", "Day", target_col}
    feature_cols = [c for c in feature_cols if c not in drop_cols]
    # Deduplicate
    feature_cols = list(dict.fromkeys(feature_cols))
    return feature_cols


# ------------------------------------------------------------------
# Step 4: Chronological split
# ------------------------------------------------------------------
def chrono_split(df: pd.DataFrame, train_frac: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


# ------------------------------------------------------------------
# Step 5: Scale features (train only)
# ------------------------------------------------------------------
def scale_features(train_df, test_df, feature_cols):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_cols].values)
    X_test = scaler.transform(test_df[feature_cols].values)
    return X_train, X_test, scaler


# ------------------------------------------------------------------
# Step 6: SVR training (with optional GridSearch)
# ------------------------------------------------------------------
def tune_and_train_svr(X_train, y_train, do_grid: bool = True):
    if not do_grid:
        model = SVR(kernel="rbf", C=10.0, epsilon=0.1)
        model.fit(X_train, y_train)
        return model, {"C": 10.0, "epsilon": 0.1, "gamma": "scale"}

    param_grid = {
        "C": [1, 10, 100],
        "epsilon": [0.01, 0.1, 0.5],
        "gamma": ["scale", 0.01, 0.1],
    }
    svr = SVR(kernel="rbf")
    grid = GridSearchCV(
        svr,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


# ------------------------------------------------------------------
# Step 7: Naive baseline (predict t+1 = value at t)
# ------------------------------------------------------------------
def naive_baseline(test_df: pd.DataFrame, target_col: str) -> np.ndarray:
    # We created lag1 during make_supervised; that's the previous day's value.
    if f"{target_col}_lag1" not in test_df.columns:
        raise RuntimeError("Lag1 column missing; check make_supervised.")
    return test_df[f"{target_col}_lag1"].values


# ------------------------------------------------------------------
# Step 8: SIR baseline — only if S, I, R present
# ------------------------------------------------------------------
def has_sir_columns(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in ["Susceptible", "Infected", "Recovered"])

def _sir_ode(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def fit_sir(train_df: pd.DataFrame, N: int):
    """
    Fit beta, gamma to the TRAIN SEGMENT using the *Infected* column.
    Assumes continuous-time ODE fit to daily sampled data.
    """
    from scipy.integrate import odeint
    from scipy.optimize import minimize

    I_data = train_df["Infected"].values.astype(float)
    S0 = float(train_df["Susceptible"].iloc[0])
    I0 = float(train_df["Infected"].iloc[0])
    R0 = float(train_df["Recovered"].iloc[0])
    t_train = np.arange(len(I_data), dtype=float)

    def loss(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return np.inf
        sol = odeint(_sir_ode, [S0, I0, R0], t_train, args=(beta, gamma, N))
        I_pred = sol[:, 1]
        return np.mean((I_data - I_pred) ** 2)

    res = minimize(loss, x0=[0.3, 0.1], bounds=[(1e-6, 2.0), (1e-6, 2.0)])
    beta_opt, gamma_opt = res.x
    return beta_opt, gamma_opt, (S0, I0, R0)

def predict_sir(beta, gamma, init_vals, N, horizon: int):
    from scipy.integrate import odeint
    t = np.arange(horizon, dtype=float)
    sol = odeint(_sir_ode, init_vals, t, args=(beta, gamma, N))
    return sol[:, 1]  # Infected predictions


# ------------------------------------------------------------------
# Step 9: Metrics helper
# ------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


# ------------------------------------------------------------------
# Step 10: Save logs
# ------------------------------------------------------------------
def save_logs(logdir, predictions_df, metrics_dict):
    os.makedirs(logdir, exist_ok=True)
    preds_path = os.path.join(logdir, "predictions.csv")
    metrics_path = os.path.join(logdir, "metrics.json")
    predictions_df.to_csv(preds_path, index=False)
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nSaved predictions -> {preds_path}")
    print(f"Saved metrics     -> {metrics_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()

    # Load data
    data = load_csv(args.data)

    # Detect target
    target_col = detect_target(data)

    # Build supervised dataset
    supervised = make_supervised(data, target_col=target_col, n_lags=args.nlags)

    # Select features (dynamic)
    feature_cols = select_features(supervised, target_col)

    # Chronological split
    train_df, test_df = chrono_split(supervised, train_frac=args.train_frac)

    # Prepare arrays
    X_train, X_test, scaler = scale_features(train_df, test_df, feature_cols)
    y_train = train_df["Target"].values
    y_test = test_df["Target"].values
    test_days = test_df["Day"].values

    # Train SVR (grid optional)
    svr_model, svr_params = tune_and_train_svr(X_train, y_train, do_grid=not args.no_grid)
    svr_preds = svr_model.predict(X_test)

    # Naive baseline
    naive_preds = naive_baseline(test_df, target_col)

    # SIR baseline (only if columns present in ORIGINAL data, not lagged table)
    sir_metrics = None
    sir_preds = None
    if has_sir_columns(data):
        # Fit SIR only on the *training portion* of original data
        # Determine train/test rows in original data consistent with supervised split
        # The supervised DF dropped first nlags rows and last row due to shifting.
        # We need to map back indices carefully.
        # Simplest: slice original data to supervised index range first.
        orig_aligned = data.iloc[-len(supervised):].reset_index(drop=True)
        orig_train = orig_aligned.iloc[:len(train_df)]
        orig_test = orig_aligned.iloc[len(train_df):]

        N = int(orig_aligned.iloc[0]["Susceptible"] + orig_aligned.iloc[0]["Infected"] + orig_aligned.iloc[0]["Recovered"])
        beta_opt, gamma_opt, init_vals = fit_sir(orig_train, N)
        sir_pred_full = predict_sir(beta_opt, gamma_opt, init_vals, N, len(orig_aligned))
        # Take only the test horizon portion
        sir_preds = sir_pred_full[len(orig_train):len(orig_aligned)]

        # Align to the *same* target as SVR/Naive comparison:
        # If target_col == "Infected", direct compare.
        # If target_col == "Reported_Cases", scale SIR infections to approx observed cases
        # using reporting ratio (mean of train_df target / train_df Infected, over aligned rows).
        if target_col != "Infected":
            # estimate reporting ratio using overlap rows
            common_len = min(len(orig_train), len(train_df))
            infected_train = orig_train["Infected"].values[:common_len]
            observed_train = train_df[target_col].values[:common_len] if target_col in train_df.columns else train_df["Target"].values[:common_len]
            ratio = observed_train.mean() / infected_train.mean() if infected_train.mean() > 0 else 0.0
            sir_preds = sir_preds * ratio

    # Metrics
    svr_metrics = compute_metrics(y_test, svr_preds)
    naive_metrics = compute_metrics(y_test, naive_preds)
    if sir_preds is not None:
        sir_metrics = compute_metrics(y_test, sir_preds)

    # Build predictions log DataFrame
    out_dict = {
        "Day": test_days,
        "Actual": y_test,
        "SVR_Pred": svr_preds,
        "Naive_Pred": naive_preds,
        "SVR_AbsErr": np.abs(y_test - svr_preds),
        "Naive_AbsErr": np.abs(y_test - naive_preds),
    }
    if sir_preds is not None:
        out_dict["SIR_Pred"] = sir_preds
        out_dict["SIR_AbsErr"] = np.abs(y_test - sir_preds)

    predictions_df = pd.DataFrame(out_dict)

    # Build metrics dict
    metrics_dict = {
        "target": target_col,
        "n_lags": args.nlags,
        "train_frac": args.train_frac,
        "features_used": feature_cols,
        "svr_params": svr_params,
        "SVR": svr_metrics,
        "Naive": naive_metrics,
    }
    if sir_metrics is not None:
        metrics_dict["SIR"] = sir_metrics
        metrics_dict["SIR_params"] = {
            "beta": float(beta_opt),
            "gamma": float(gamma_opt),
        }

    # Save
    save_logs(args.logdir, predictions_df, metrics_dict)

    # Minimal console summary (no plots)
    print("\n=== Summary ===")
    print(f"Target column: {target_col}")
    print(f"Rows train/test: {len(train_df)}/{len(test_df)}")
    print("SVR metrics:", svr_metrics)
    print("Naive metrics:", naive_metrics)
    if sir_metrics is not None:
        print("SIR metrics:", sir_metrics)


if __name__ == "__main__":
    main()
