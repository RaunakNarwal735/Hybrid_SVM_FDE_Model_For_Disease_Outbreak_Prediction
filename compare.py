import argparse
import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from joblib import load
from datetime import datetime

from models.utils import chrono_split, compute_metrics
from models.svr_model import scale_features, tune_and_train_svr
from models.sir_baseline import has_sir_columns, fit_sir, predict_sir


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare models (Naive, SVR, SIR, SOUP-S1, Naive-SVR).")
    parser.add_argument("--data_dir", required=True, help="Path to folder containing datasets.")
    # Default directories as per user request
    parser.add_argument("--soup_dir", default="C:\\Users\\rishu narwal\\Desktop\\SVM_FDE\\trained\\20",
                        help="Path to SOUP-S1 trained model.")
    parser.add_argument("--naive_svr_dir", default="C:\\Users\\rishu narwal\\Desktop\\SVM_FDE\\trained\\Naive_SVR",
                        help="Path to Naive-SVR trained model.")
    parser.add_argument("--out_dir", default="C:\\Users\\rishu narwal\\Desktop\\SVM_FDE\\logs",
                        help="Directory to save logs and plots.")
    parser.add_argument("--nlags", type=int, default=3, help="Number of lags for supervised framing.")
    parser.add_argument("--train_frac", type=float, default=0.8, help="Train/test split fraction.")
    parser.add_argument("--no_grid", action="store_true", help="Disable GridSearchCV for local SVR baseline.")
    parser.add_argument("--save_plots", action="store_true", help="Save per-dataset plots.")
    return parser.parse_args()


def make_supervised(df, target_col="Reported_Cases", n_lags=3):
    out = df.copy()
    for lag in range(1, n_lags + 1):
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    out["Naive"] = out[target_col].shift(1)  # baseline
    out["Target"] = out[target_col].shift(-1)
    return out.dropna().reset_index(drop=True)


def plot_predictions(days, actual, sir_pred, soup_pred, svr_pred, naive_pred, naive_svr_pred, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(days, actual, label="Actual", color="blue")
    if sir_pred is not None:
        plt.plot(days, sir_pred, label="SIR", color="orange", linestyle="-.")
    if soup_pred is not None:
        plt.plot(days, soup_pred, label="SOUP-S1", color="purple")
    if svr_pred is not None:
        plt.plot(days, svr_pred, label="SVR", color="red", linestyle="--")
    if naive_pred is not None:
        plt.plot(days, naive_pred, label="Naive", color="green", linestyle=":")
    if naive_svr_pred is not None:
        plt.plot(days, naive_svr_pred, label="Naive-SVR", color="brown", linestyle="-.")
    plt.title("Model Comparison")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------
# Evaluate Single Dataset
# ------------------------------------------------------------
def evaluate_single_dataset(csv_path, soup_model, soup_scaler, soup_features,
                            naive_svr_model, naive_svr_scaler, naive_svr_features,
                            out_dir, nlags=3, train_frac=0.8, no_grid=False, save_plots=False):
    df = pd.read_csv(csv_path)
    if "Day" not in df.columns:
        df.insert(0, "Day", np.arange(len(df)))

    target_col = "Reported_Cases" if "Reported_Cases" in df.columns else "Infected"
    supervised = make_supervised(df, target_col, nlags)  # creates Naive + lag cols + Target

    # --------------------------------------------------
    # SIR Baseline (full curve) -> align to supervised rows
    # --------------------------------------------------
    if has_sir_columns(df):
        N = int(df[["Susceptible", "Infected", "Recovered"]].iloc[0].sum())
        beta, gamma, init_vals = fit_sir(df, N)
        sir_curve = predict_sir(beta, gamma, init_vals, N, len(df))  # predicted *infecteds*

        # scale to target if needed
        if target_col != "Infected":
            tgt = df[target_col].values.astype(float)
            denom = df["Infected"].replace(0, np.nan).values.astype(float)
            ratio = np.nanmean(tgt / denom)
            if not np.isfinite(ratio) or ratio <= 0:
                ratio = tgt.mean() / max(df["Infected"].mean(), 1e-6)
            sir_curve = sir_curve * ratio

        # supervised rows predict next-day target -> grab sir_curve[t+1]
        idx_plus1 = supervised["Day"].values.astype(int) + 1
        idx_plus1[idx_plus1 >= len(sir_curve)] = len(sir_curve) - 1
        supervised["SIR_Base"] = sir_curve[idx_plus1]
    else:
        # no SIR available
        supervised["SIR_Base"] = np.nan

    # --------------------------------------------------
    # Ensure feature cols present
    # --------------------------------------------------
    if soup_features:
        for col in soup_features:
            if col not in supervised.columns:
                supervised[col] = 0.0
    if naive_svr_features:
        for col in naive_svr_features:
            if col not in supervised.columns:
                supervised[col] = 0.0

    # --------------------------------------------------
    # Split
    # --------------------------------------------------
    train_df, test_df = chrono_split(supervised, train_frac)
    y_train = train_df["Target"].values
    y_test = test_df["Target"].values
    test_days = test_df["Day"].values

    # --------------------------------------------------
    # SVR baseline (local, no SIR_Base)
    # --------------------------------------------------
    local_feats = [c for c in supervised.columns if c.startswith(f"{target_col}_lag")]
    X_train, X_test, _ = scale_features(train_df, test_df, local_feats)
    local_svr_model, local_params = tune_and_train_svr(X_train, y_train, grid_search=not no_grid)
    svr_preds = np.clip(local_svr_model.predict(X_test), 0, None)

    # --------------------------------------------------
    # Naive baseline
    # --------------------------------------------------
    # we built Naive in make_supervised() -> lag-1 raw
    naive_preds = np.roll(y_test, 1)
    naive_preds[0] = y_train[-1]

    # --------------------------------------------------
    # SIR test preds & metrics
    # --------------------------------------------------
    if test_df["SIR_Base"].notna().any():
        sir_preds = test_df["SIR_Base"].fillna(0).values
        sir_metrics = compute_metrics(y_test, sir_preds)
    else:
        sir_preds = None
        sir_metrics = None

    # --------------------------------------------------
    # SOUP-S1 (uses pre-trained hybrid)
    # --------------------------------------------------
    if soup_model is not None:
        soup_X_test = test_df[soup_features].fillna(0.0).values
        soup_X_test_scaled = soup_scaler.transform(soup_X_test)
        residual_preds = soup_model.predict(soup_X_test_scaled)
        if sir_preds is not None:
            soup_preds = np.clip(sir_preds + residual_preds, 0, None)
        else:
            soup_preds = np.clip(residual_preds, 0, None)
        soup_metrics = compute_metrics(y_test, soup_preds)
    else:
        soup_preds = None
        soup_metrics = None

    # --------------------------------------------------
    # Naive-SVR (pre-trained)
    # --------------------------------------------------
    if naive_svr_model is not None:
        ns_cols = [c for c in naive_svr_features if c in test_df.columns]
        ns_X_test = test_df[ns_cols].fillna(0.0).values
        ns_X_test_scaled = naive_svr_scaler.transform(ns_X_test)
        ns_resid = naive_svr_model.predict(ns_X_test_scaled)
        # add naive baseline (lag-1 actual)
        # careful: test_df["Naive"] is today's naive pred for Target? We built Naive=lag1 raw,
        # Target is t+1 => naive for Target is current day's actual -> shift align is correct.
        naive_base_for_target = test_df["Naive"].values
        naive_svr_preds = np.clip(naive_base_for_target + ns_resid, 0, None)
        naive_svr_metrics = compute_metrics(y_test, naive_svr_preds)
    else:
        naive_svr_preds = None
        naive_svr_metrics = None

    # --------------------------------------------------
    # Collect metrics
    # --------------------------------------------------
    metrics = {
        "Dataset_ID": os.path.basename(os.path.dirname(csv_path)),
        "SVR": compute_metrics(y_test, svr_preds),
        "Naive": compute_metrics(y_test, naive_preds),
        "SOUP-S1": soup_metrics if soup_metrics else {},
        "Naive-SVR": naive_svr_metrics if naive_svr_metrics else {},
    }
    if sir_metrics:
        metrics["SIR"] = sir_metrics

    # --------------------------------------------------
    # Save outputs
    # --------------------------------------------------
    ds_dir = os.path.join(out_dir, os.path.basename(os.path.dirname(csv_path)))
    os.makedirs(ds_dir, exist_ok=True)

    preds_df = pd.DataFrame({
        "Day": test_days,
        "Actual": y_test,
        "Naive": naive_preds,
        "SVR": svr_preds,
        "SOUP-S1": soup_preds if soup_preds is not None else [None] * len(y_test),
        "Naive-SVR": naive_svr_preds if naive_svr_preds is not None else [None] * len(y_test),
        "SIR": sir_preds if sir_preds is not None else [None] * len(y_test),
    })
    preds_df.to_csv(os.path.join(ds_dir, "predictions.csv"), index=False)

    with open(os.path.join(ds_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    if save_plots:
        plot_predictions(
            test_days, y_test,
            sir_preds, soup_preds, svr_preds, naive_preds, naive_svr_preds,
            os.path.join(ds_dir, "plot.png")
        )

    return metrics



# ------------------------------------------------------------
# Error Summary Plot
# ------------------------------------------------------------
def plot_error_summary(metrics_df, out_path):
    models = ["Naive", "SVR", "SIR", "SOUP-S1", "Naive-SVR"]
    metrics = ["RMSE", "MAE", "R2"]

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()

    # Plot RMSE, MAE, R2
    for idx, metric in enumerate(metrics):
        ax = axs[idx]
        for model in models:
            col = f"{model}_{metric}"
            if col in metrics_df.columns:
                ax.plot(metrics_df["Dataset_ID"], metrics_df[col], marker="o", label=model)
        ax.set_title(f"{metric} Comparison")
        ax.set_xticks(range(len(metrics_df)))
        ax.set_xticklabels(metrics_df["Dataset_ID"], rotation=45, ha="right")
        ax.legend()
        ax.grid(True)

    # Average metrics
    ax = axs[3]
    avg_values = {m: {} for m in models}
    for model in models:
        for metric in metrics:
            col = f"{model}_{metric}"
            avg_values[model][metric] = metrics_df[col].mean(skipna=True) if col in metrics_df else np.nan

    bar_width = 0.2
    x = np.arange(len(models))
    ax.bar(x - bar_width, [avg_values[m]["RMSE"] for m in models], width=bar_width, label="Avg RMSE")
    ax.bar(x, [avg_values[m]["MAE"] for m in models], width=bar_width, label="Avg MAE")
    ax.bar(x + bar_width, [avg_values[m]["R2"] for m in models], width=bar_width, label="Avg R²")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Average Metrics Across Models")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load SOUP-S1 model
    soup_model, soup_scaler, soup_features = None, None, None
    if os.path.exists(os.path.join(args.soup_dir, "hybrid_svr.joblib")):
        print("Loading SOUP-S1 model...")
        soup_model = load(os.path.join(args.soup_dir, "hybrid_svr.joblib"))
        soup_scaler = load(os.path.join(args.soup_dir, "scaler.joblib"))
        with open(os.path.join(args.soup_dir, "training_metadata.json"), "r") as f:
            soup_features = json.load(f)["feature_cols"]

    # Load Naive-SVR model
    naive_svr_model, naive_svr_scaler, naive_svr_features = None, None, None
    if os.path.exists(os.path.join(args.naive_svr_dir, "naive_svr.joblib")):
        print("Loading Naive-SVR model...")
        naive_svr_model = load(os.path.join(args.naive_svr_dir, "naive_svr.joblib"))
        naive_svr_scaler = load(os.path.join(args.naive_svr_dir, "scaler.joblib"))
        with open(os.path.join(args.naive_svr_dir, "training_metadata.json"), "r") as f:
            naive_svr_features = json.load(f)["feature_cols"]

    # Find all datasets
    csv_files = glob.glob(os.path.join(args.data_dir, "**", "dataset.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No dataset.csv found in {args.data_dir}")

    print(f"Found {len(csv_files)} datasets. Evaluating...")

    all_metrics = []
    for csv_path in csv_files:
        print(f" -> {csv_path}")
        metrics = evaluate_single_dataset(csv_path, soup_model, soup_scaler, soup_features,
                                          naive_svr_model, naive_svr_scaler, naive_svr_features,
                                          args.out_dir, nlags=args.nlags,
                                          train_frac=args.train_frac, no_grid=args.no_grid,
                                          save_plots=args.save_plots)
        all_metrics.append(metrics)

    # Aggregate metrics
    metrics_flat = []
    for row in all_metrics:
        flat = {"Dataset_ID": row["Dataset_ID"]}
        for m in ["Naive", "SVR", "SIR", "SOUP-S1", "Naive-SVR"]:
            if m in row:
                for k, v in row[m].items():
                    flat[f"{m}_{k}"] = v
        metrics_flat.append(flat)

    metrics_df = pd.DataFrame(metrics_flat)
    metrics_csv_path = os.path.join(args.out_dir, "all_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Final error chart
    final_chart = os.path.join(args.out_dir, "error_comparison_summary.png")
    plot_error_summary(metrics_df, final_chart)

    print(f"\nComparison complete.")
    print(f"Results saved in: {args.out_dir}")
    print(f"Final summary chart: {final_chart}")


if __name__ == "__main__":
    main()
