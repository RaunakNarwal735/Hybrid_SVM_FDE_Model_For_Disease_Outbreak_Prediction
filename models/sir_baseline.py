import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize

def has_sir_columns(df):
    """
    Check if DataFrame contains columns required for SIR baseline.
    
    Args:
        df (pd.DataFrame): Input dataset.
    
    Returns:
        bool: True if Susceptible, Infected, Recovered columns exist.
    """
    required_cols = {"Susceptible", "Infected", "Recovered"}
    return required_cols.issubset(df.columns)

def sir_ode(y, t, beta, gamma, N):
    """
    Differential equations for SIR model.
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def fit_sir(train_df, N):
    """
    Fit beta and gamma parameters for SIR model to training data.
    
    Args:
        train_df (pd.DataFrame): Training portion of data.
        N (int): Total population.
    
    Returns:
        tuple: (beta, gamma, (S0, I0, R0))
    """
    I_data = train_df["Infected"].values.astype(float)
    S0, I0, R0 = train_df.iloc[0][["Susceptible", "Infected", "Recovered"]]
    t_train = np.arange(len(I_data), dtype=float)

    def loss(params):
        beta, gamma = params
        if beta <= 0 or gamma <= 0:
            return np.inf
        sol = odeint(sir_ode, [S0, I0, R0], t_train, args=(beta, gamma, N))
        I_pred = sol[:, 1]
        return np.mean((I_data - I_pred) ** 2)

    res = minimize(loss, x0=[0.3, 0.1], bounds=[(1e-6, 2.0), (1e-6, 2.0)])
    beta_opt, gamma_opt = res.x
    return beta_opt, gamma_opt, (S0, I0, R0)

def predict_sir(beta, gamma, init_vals, N, horizon):
    """
    Predict infections for given horizon using fitted SIR parameters.
    
    Args:
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        init_vals (tuple): Initial values (S0, I0, R0).
        N (int): Total population.
        horizon (int): Number of future days to predict.
    
    Returns:
        np.ndarray: Predicted infections for each day.
    """
    t = np.arange(horizon, dtype=float)
    sol = odeint(sir_ode, init_vals, t, args=(beta, gamma, N))
    return sol[:, 1]  # Infected compartment
