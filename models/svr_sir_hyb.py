import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import curve_fit
import json

# --- 1. SIR Fit ---
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
    return beta_opt, gamma_opt, sir_curve(t, beta_opt, gamma_opt)

# --- 2. Feature Creation ---
def create_features(data, lags=7):
    df = data.copy()
    # Create lags of Reported_Cases
    for lag in range(1, lags + 1):
        df[f"Reported_Cases_lag{lag}"] = df["Reported_Cases"].shift(lag)
    df = df.dropna()
    return df

# --- 3. Hybrid Model ---
def train_hybrid_sir_svr(data_path, log_path="hybrid_metrics.json", lags=7):
    data = pd.read_csv(data_path)
    population = data[["Susceptible", "Infected", "Recovered"]].iloc[0].sum()
    y_cases = data["Reported_Cases"].values

    # 1. Fit SIR
    beta, gamma, sir_pred = fit_sir(y_cases, N=population)
    residuals = y_cases - sir_pred
    residuals = np.clip(residuals, 0, None)

    # 2. Add residual column for training
    data["Residual"] = residuals
    df = create_features(data, lags=lags)

    # 3. Build features for SVR (use all columns except target)
    feature_cols = [c for c in df.columns if c not in ["Reported_Cases", "Residual"]]
    X = df[feature_cols].values
    y = df["Residual"].values

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 5. Train SVR
    svr = SVR(kernel="rbf", C=100, epsilon=0.1)
    svr.fit(X_train, y_train)
    residual_pred = svr.predict(X_test)

    # 6. Final prediction = SIR + SVR residual
    sir_shifted = sir_pred[-len(residual_pred):]
    final_pred = np.clip(sir_shifted + residual_pred, 0, None)

    # 7. Evaluate
    true = y_cases[-len(final_pred):]
    mse = mean_squared_error(true, final_pred)
    mae = mean_absolute_error(true, final_pred)
    r2 = r2_score(true, final_pred)

    metrics = {
        "beta": float(beta),
        "gamma": float(gamma),
        "RMSE": np.sqrt(mse),
        "MAE": mae,
        "R2": r2,
        "features_used": feature_cols
    }

    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Hybrid SIR-SVR Metrics: {metrics}")

    return final_pred, metrics, svr
