import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime

def chrono_split(df, train_frac=0.8):
    """
    Split DataFrame into train and test in chronological order.
    
    Args:
        df (pd.DataFrame): Full dataset.
        train_frac (float): Fraction for training (0 < train_frac < 1).
    
    Returns:
        (pd.DataFrame, pd.DataFrame): train_df, test_df
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1")
    split_idx = int(len(df) * train_frac)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def compute_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, R² metrics.
    
    Args:
        y_true (array-like): Actual values.
        y_pred (array-like): Predicted values.
    
    Returns:
        dict: {"RMSE": float, "MAE": float, "R2": float}
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}



def _timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_logs(logdir, predictions_df, metrics_dict):
    """
    Save predictions and metrics to CSV and JSON, using timestamped filenames to
    avoid Windows file-lock issues (e.g., file open in Excel).
    """
    os.makedirs(logdir, exist_ok=True)

    ts = _timestamp()
    preds_path = os.path.join(logdir, f"predictions_{ts}.csv")
    metrics_path = os.path.join(logdir, f"metrics_{ts}.json")

    try:
        predictions_df.to_csv(preds_path, index=False)
    except PermissionError:
        # Fallback: write with random suffix
        alt_path = os.path.join(logdir, f"predictions_{ts}_alt.csv")
        predictions_df.to_csv(alt_path, index=False)
        preds_path = alt_path

    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
    except PermissionError:
        alt_path = os.path.join(logdir, f"metrics_{ts}_alt.json")
        with open(alt_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)
        metrics_path = alt_path

    print(f"Logs saved:\n  {preds_path}\n  {metrics_path}")

