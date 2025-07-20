import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from models.utils import chrono_split, compute_metrics

# ------------------------------------------------------------
# Prepare supervised data for Naive-SVR
# ------------------------------------------------------------
def make_supervised(df, target_col="Reported_Cases", n_lags=3, dataset_id=""):
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    df["Naive"] = df[target_col].shift(1)
    df["Residual"] = df[target_col] - df["Naive"]
    df["Target"] = df["Residual"].shift(-1)
    df["Dataset_ID"] = dataset_id
    return df.dropna().reset_index(drop=True)

# ------------------------------------------------------------
# Train Naive-SVR model
# ------------------------------------------------------------
def train_naive_svr_batch(data_dir, target_col="Reported_Cases", n_lags=3, train_frac=0.8):
    all_data = []
    csv_files = glob.glob(os.path.join(data_dir, "**", "dataset.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No dataset.csv found in {data_dir}")

    print(f"Found {len(csv_files)} datasets. Preparing data...")

    for csv_path in csv_files:
        dataset_id = os.path.basename(os.path.dirname(csv_path))
        df = pd.read_csv(csv_path)
        supervised = make_supervised(df, target_col, n_lags, dataset_id)
        all_data.append(supervised)

    big_df = pd.concat(all_data, ignore_index=True)
    features = [c for c in big_df.columns if c.startswith(f"{target_col}_lag")]

    train_df, test_df = chrono_split(big_df, train_frac)
    X_train, X_test = train_df[features].values, test_df[features].values
    y_train, y_test = train_df["Target"].values, test_df["Target"].values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train SVR on residuals
    param_grid = {"C": [1, 10, 100], "epsilon": [0.1, 0.5, 1.0], "gamma": [0.01, 0.1, 1.0]}
    svr = GridSearchCV(SVR(kernel="rbf"), param_grid, cv=3, n_jobs=-1)
    svr.fit(X_train, y_train)
    print(f"Best SVR params: {svr.best_params_}")

    preds = np.clip(svr.predict(X_test) + test_df["Naive"].values, 0, None)
    metrics = compute_metrics(test_df[target_col].values, preds)
    print(f"Naive-SVR Metrics: {metrics}")

    return svr.best_estimator_, scaler, features, metrics

# ------------------------------------------------------------
# Main entry
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to datasets folder (batch).")
    parser.add_argument("--out_dir", default="naive_svr_trained", help="Directory to save model artifacts.")
    parser.add_argument("--nlags", type=int, default=3, help="Number of lag features.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, scaler, features, metrics = train_naive_svr_batch(args.data_dir, n_lags=args.nlags)

    dump(model, os.path.join(args.out_dir, "naive_svr.joblib"))
    dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))
    with open(os.path.join(args.out_dir, "training_metadata.json"), "w") as f:
        json.dump({"feature_cols": features, "metrics": metrics}, f, indent=4)

    print(f"Model saved in {args.out_dir}")

if __name__ == "__main__":
    main()
