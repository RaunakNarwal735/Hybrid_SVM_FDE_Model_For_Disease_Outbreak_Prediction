import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_epidemic_data(N=10000, beta0=0.3, gamma=0.1, I0=10, days=180,
                            seasonal_amp=0.25, seasonal_period=60,
                            interventions=[(70,100,0.4)], reporting_start=0.6, reporting_end=0.85,
                            noise=True):
    S, I, R = N - I0, I0, 0
    S_list, I_list, R_list = [S], [I], [R]
    reported_cases_list, beta_list, season_idx_list, int_flag_list = [], [], [], []

    report_probs = np.linspace(reporting_start, reporting_end, days)

    for t in range(1, days):
        # Seasonal effect on beta
        season_factor = 1 + seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)
        beta = beta0 * season_factor

        # Intervention effect
        int_flag = 0
        for (start, end, mult) in interventions:
            if start <= t <= end:
                beta *= mult
                int_flag = 1

        # SIR dynamics
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I

        S += dS
        I += dI
        R += dR

        # Noise: reported cases ~ Poisson(true new infections * reporting probability)
        true_new_infections = max(beta * S * I / N, 0)
        report_prob = report_probs[t]
        reported_cases = np.random.poisson(true_new_infections * report_prob) if noise else true_new_infections * report_prob

        # Store values
        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        reported_cases_list.append(reported_cases)
        beta_list.append(beta)
        season_idx_list.append(season_factor)
        int_flag_list.append(int_flag)

    df = pd.DataFrame({
        "Day": np.arange(days),
        "Susceptible": S_list,
        "Infected": I_list,
        "Recovered": R_list,
        "Reported_Cases": [0] + reported_cases_list,
        "Beta_Effective": [beta0] + beta_list,
        "Season_Index": [1] + season_idx_list,
        "Intervention_Flag": [0] + int_flag_list
    })
    return df

# Generate data
data = generate_epidemic_data()

# Plot stacked SIR
plt.figure(figsize=(10,6))
plt.stackplot(data["Day"],  data["Infected"], data["Susceptible"],data["Recovered"], labels=[ 'Infected','Susceptible', 'Recovered'], colors=['#F6BE00','#008B8B', '#353E43'])
plt.xlim(0, max(data["Day"]))
plt.ylim(0, data["Susceptible"].iloc[0] + data["Infected"].iloc[0] + data["Recovered"].iloc[0])
plt.title("Epidemic Dynamics with Seasonality & Interventions", fontsize=16)
plt.xlabel("Time (days)")
plt.ylabel("Population")
plt.legend()
plt.show()

# Plot reported cases
plt.figure(figsize=(10,5))
plt.plot(data["Day"], data["Reported_Cases"], color='purple', label="Reported Cases")
plt.xlabel("Time (days)")
plt.ylabel("Reported Cases")
plt.title("Simulated Reported Cases (Noisy Observations)")
plt.legend()
plt.show()
