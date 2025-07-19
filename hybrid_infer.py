import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# SIR MODEL
# -------------------------------------------------
def sir_model(t, beta, gamma, N, I0, R0):
    S, I, R = [N - I0 - R0], [I0], [R0]
    for _ in range(1, len(t)):
        S_new = S[-1] - beta * S[-1] * I[-1] / N
        I_new = I[-1] + beta * S[-1] * I[-1] / N - gamma * I[-1]
        R_new = R[-1] + gamma * I[-1]
        S.append(S_new)
        I.append(I_new)
        R.append(R_new)
    return np.array(I)

def fit_sir(y_cases, N):
    t = np.arange(len(y_cases))
    I0 = max(y_cases[0], 1)
    R0 = 0
    def sir_curve(t, beta, gamma):
        return sir_model(t, beta, gamma, N, I0, R0)
    popt, _ = curve_fit(sir_curve, t, y_cases, p0=[0.2, 0.1], bounds=(0, [1.0, 1.0]))
    beta_opt, gamma_opt = popt
    I_pred = sir_curve(t, beta_opt, gamma_opt)
    return beta_opt, gamma_opt, I_pred

# -------------------------------------------------
# FEATURE CREATION
# -------------------------------------------------
def create_lagged_features(df, lags=7):
    for lag in range(1, lags + 1):
        df[f"Reported_Cases_lag{lag}"] = df["Reported_Cases"].shift(lag)
    return df.dropna()

def detect_population(df):
    cols = ["Susceptible", "Infected", "Recovered"]
    if all(c in df.columns for c in cols):
        return df[cols].iloc[0].sum()
    elif "Susceptible" in df.columns:
        return df["Susceptible"].iloc[0]
    elif "Infected" in df.columns:
        return df["Infected"].iloc[0] * 10  # fallback
    else:
        return df["Reported_Cases"].max() * 50  # fallback heuristic

# -------------------------------------------------
# MAIN INFERENCE FUNCTION
# -------------------------------------------------
def run_inference(data_path, model_dir, output_path=None):
    # Load data
    df = pd.read_csv(data_path)
    if "Reported_Cases" not in df.columns:
        raise ValueError("Dataset must have a 'Reported_Cases' column.")

    # Population detection
    population = detect_population(df)
    print(f"[INFO] Population detected: {population}")

    # Fit SIR
    beta, gamma, sir_pred = fit_sir(df["Reported_Cases"].values, N=population)
    df["SIR_Base"] = sir_pred

    # Add lags and drop NA
    df_feat = create_lagged_features(df.copy(), lags=7)
    print(f"[INFO] Dataset shape after lagging: {df_feat.shape}")

    # Load trained model & scaler
    svr = load(os.path.join(model_dir, "hybrid_svr.joblib"))
    scaler = load(os.path.join(model_dir, "scaler.joblib"))
    with open(os.path.join(model_dir, "training_metadata.json"), "r") as f:
        metadata = json.load(f)
    feature_cols = metadata["feature_cols"]

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = 0  # fill missing

    X = df_feat[feature_cols].fillna(0.0).values
    X_scaled = scaler.transform(X)

    # Predict residuals
    residual_pred = svr.predict(X_scaled)
    df_feat["Hybrid_Pred"] = np.clip(df_feat["SIR_Base"] + residual_pred, 0, None)

    # Save predictions
    if output_path is None:
        output_path = os.path.join(model_dir, "inference_output.csv")
    df_feat.to_csv(output_path, index=False)
    print(f"[INFO] Predictions saved at {output_path}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df_feat["Reported_Cases"].values, label="Actual")
    plt.plot(df_feat["SIR_Base"].values, label="SIR Base")
    plt.plot(df_feat["Hybrid_Pred"].values, label="Hybrid")
    plt.legend()
    plt.title("Hybrid vs SIR vs Actual (Inference)")
    plt.xlabel("Days")
    plt.ylabel("Reported Cases")
    plt.savefig(os.path.join(model_dir, "inference_plot.png"))
    plt.show()

    return df_feat

# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid SIR-SVR Inference")
    parser.add_argument("--data", required=True, help="Path to dataset CSV file.")
    parser.add_argument("--model_dir", required=True, help="Directory containing hybrid_svr.joblib and scaler.joblib.")
    parser.add_argument("--output", default=None, help="Output CSV path for predictions.")
    args = parser.parse_args()

    run_inference(args.data, args.model_dir, args.output)

