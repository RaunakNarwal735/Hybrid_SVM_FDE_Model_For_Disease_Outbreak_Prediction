import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
from models.utils import chrono_split, compute_metrics, save_logs
from models.svr_model import scale_features, tune_and_train_svr
from models.sir_baseline import has_sir_columns, fit_sir, predict_sir


def parse_args():
    parser = argparse.ArgumentParser(description="Run SVR, Naive, and SIR baselines for epidemic forecasting.")
    parser.add_argument("--data", required=True, help="Path to input CSV dataset.")
    parser.add_argument("--logdir", default="logs", help="Directory to save logs and plots.")
    parser.add_argument("--nlags", type=int, default=3, help="Number of lag features.")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Fraction of rows used for training.")
    parser.add_argument("--no_grid", action="store_true", help="Disable GridSearchCV for SVR.")
    return parser.parse_args()


def make_supervised(df, target_col="Reported_Cases", n_lags=3):
    out = df.copy()
    for lag in range(1, n_lags + 1):
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    out["Target"] = out[target_col].shift(-1)
    return out.dropna().reset_index(drop=True)


def select_features(df, target_col):
    feature_cols = [c for c in df.columns if "lag" in c]
    optional = ["Beta_Effective", "Season_Index", "Intervention_Flag", "Reporting_Prob"]
    for col in optional:
        if col in df.columns:
            feature_cols.append(col)
    return list(dict.fromkeys(feature_cols))


def timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_predictions(days, actual, svr_pred, naive_pred, sir_pred, plotdir):
    os.makedirs(plotdir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(days, actual, label="Actual", color="blue")
    plt.plot(days, svr_pred, label="SVR", color="red", linestyle="--")
    plt.plot(days, naive_pred, label="Naive", color="green", linestyle=":")
    if sir_pred is not None:
        plt.plot(days, sir_pred, label="SIR", color="orange", linestyle="-.")
    plt.title("Predictions vs Actual")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.legend()

    ts = timestamp()
    plot_path = os.path.join(plotdir, f"predictions_plot_{ts}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path



def main():
    args = parse_args()

    # Load dataset
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset not found: {args.data}")

    df = pd.read_csv(args.data)
    if "Day" not in df.columns:
        df.insert(0, "Day", np.arange(len(df)))

    # Detect target column
    if "Reported_Cases" in df.columns:
        target_col = "Reported_Cases"
    elif "Infected" in df.columns:
        target_col = "Infected"
    else:
        raise ValueError("Dataset must contain 'Reported_Cases' or 'Infected' column.")

    # Prepare supervised data
    supervised = make_supervised(df, target_col, args.nlags)
    feature_cols = select_features(supervised, target_col)

    # Split into train/test
    train_df, test_df = chrono_split(supervised, args.train_frac)
    X_train, X_test, _ = scale_features(train_df, test_df, feature_cols)
    y_train, y_test = train_df["Target"].values, test_df["Target"].values
    test_days = test_df["Day"].values

    # Train SVR
    svr_model, svr_params = tune_and_train_svr(X_train, y_train, grid_search=not args.no_grid)
    svr_preds = svr_model.predict(X_test)
    svr_preds = np.clip(svr_preds, 0, None)  # ✅ Prevent negatives

    # Naive baseline (corrected to avoid leakage)
    naive_preds = np.roll(y_test, 1)
    naive_preds[0] = y_train[-1]

    # SIR baseline
    sir_preds = None
    sir_metrics = None
    if has_sir_columns(df):
        aligned = df.iloc[-len(supervised):].reset_index(drop=True)
        orig_train = aligned.iloc[:len(train_df)]
        N = int(aligned.iloc[0][["Susceptible", "Infected", "Recovered"]].sum())
        beta, gamma, init_vals = fit_sir(orig_train, N)
        full_sir_pred = predict_sir(beta, gamma, init_vals, N, len(aligned))
        sir_preds = full_sir_pred[len(orig_train):]
        if target_col != "Infected":
            ratio = y_train.mean() / orig_train["Infected"].mean()
            sir_preds *= ratio
        sir_metrics = compute_metrics(y_test, sir_preds)

    # Compute metrics
    svr_metrics = compute_metrics(y_test, svr_preds)
    naive_metrics = compute_metrics(y_test, naive_preds)

    # Save predictions
    results = {
        "Day": test_days,
        "Actual": y_test,
        "SVR_Pred": svr_preds,
        "Naive_Pred": naive_preds,
        "SVR_AbsErr": np.abs(y_test - svr_preds),
        "Naive_AbsErr": np.abs(y_test - naive_preds),
    }
    if sir_preds is not None:
        results["SIR_Pred"] = sir_preds
        results["SIR_AbsErr"] = np.abs(y_test - sir_preds)
    predictions_df = pd.DataFrame(results)

    # Prepare metrics log
    metrics_dict = {
        "target": target_col,
        "features": feature_cols,
        "svr_params": svr_params,
        "SVR": svr_metrics,
        "Naive": naive_metrics,
        "notes": ["SVR predictions clipped to >= 0", "Naive baseline corrected"]
    }
    if sir_metrics:
        metrics_dict["SIR"] = sir_metrics
        metrics_dict["SIR_params"] = {"beta": float(beta), "gamma": float(gamma)}

    # Save logs
    save_logs(args.logdir, predictions_df, metrics_dict)

    # Save plot
    plot_path = plot_predictions(test_days, y_test, svr_preds, naive_preds, sir_preds, "plots")


    # Print summary
    print("\n=== Summary ===")
    print(f"Target: {target_col}")
    print(f"Train/Test: {len(train_df)}/{len(test_df)}")
    print("SVR:", svr_metrics)
    print("Naive:", naive_metrics)
    if sir_metrics:
        print("SIR:", sir_metrics)
    print(f"Plot saved at: {plot_path}")


if __name__ == "__main__":
    main()
