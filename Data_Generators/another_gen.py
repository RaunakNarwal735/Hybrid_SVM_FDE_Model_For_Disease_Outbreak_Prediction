import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_epidemic_data(
    N=10000,
    beta0=0.3,
    gamma=0.1,
    I0=10,
    days=180,
    seasonal_amp=0.25,
    seasonal_period=60,
    seasonal_phase=0.0,
    interventions=[(70, 100, 0.4)],  # (start_day, end_day, multiplier on beta)
    reporting_start=0.6,
    reporting_end=0.85,
    random_seed=42,
):
    """
    Generate integer-based stochastic SIR data with seasonality, interventions, and reporting noise.
    Returns a DataFrame with true compartments and noisy reported cases.
    """
    rng = np.random.default_rng(random_seed)

    report_probs = np.linspace(reporting_start, reporting_end, days)
    S, I, R = N - I0, I0, 0

    S_list, I_list, R_list = [S], [I], [R]
    reported_cases_list = [0]
    beta_eff_list, season_idx_list, int_flag_list = [beta0], [1.0], [0]
    new_inf_list, new_rec_list = [0], [0]

    for t in range(1, days):
        season_factor = 1.0 + seasonal_amp * np.sin((2 * np.pi * t / seasonal_period) + seasonal_phase)
        int_flag = 0
        beta_t = beta0 * season_factor
        for (start, end, mult) in interventions:
            if start <= t <= end:
                beta_t *= mult
                int_flag = 1

        p_inf = 1.0 - np.exp(-beta_t * (I / N))
        p_inf = np.clip(p_inf, 0.0, 1.0)
        p_rec = 1.0 - np.exp(-gamma)
        p_rec = np.clip(p_rec, 0.0, 1.0)

        new_inf = rng.binomial(S, p_inf) if S > 0 else 0
        new_rec = rng.binomial(I, p_rec) if I > 0 else 0

        S -= new_inf
        I += new_inf - new_rec
        R += new_rec

        report_p = report_probs[t]
        reported_cases = rng.binomial(new_inf, report_p) if new_inf > 0 else 0

        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        reported_cases_list.append(reported_cases)
        beta_eff_list.append(beta_t)
        season_idx_list.append(season_factor)
        int_flag_list.append(int_flag)
        new_inf_list.append(new_inf)
        new_rec_list.append(new_rec)

    df = pd.DataFrame({
        "Day": np.arange(days),
        "Susceptible": S_list,
        "Infected": I_list,
        "Recovered": R_list,
        "New_Infections": new_inf_list,
        "New_Recoveries": new_rec_list,
        "Reported_Cases": reported_cases_list,
        "Beta_Effective": beta_eff_list,
        "Season_Index": season_idx_list,
        "Intervention_Flag": int_flag_list,
        "Reporting_Prob": report_probs
    })
    return df


def save_dataset(df, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"Saved dataset -> {filepath}")


def save_plots(df, outdir="plots", prefix="simulation"):
    os.makedirs(outdir, exist_ok=True)

    # Stacked SIR plot
    plt.figure(figsize=(10, 6))
    plt.stackplot(df["Day"], df["Infected"], df["Susceptible"], df["Recovered"],
                  labels=['Infected', 'Susceptible', 'Recovered'],
                  colors=['#F6BE00', '#008B8B', '#353E43'])
    plt.xlim(0, max(df["Day"]))
    plt.ylim(0, df["Susceptible"].iloc[0] + df["Infected"].iloc[0] + df["Recovered"].iloc[0])
    plt.title("Stochastic Epidemic with Seasonality & Interventions")
    plt.xlabel("Day")
    plt.ylabel("People")
    plt.legend(loc="upper right")
    plt.tight_layout()
    sir_path = os.path.join(outdir, f"{prefix}_sir_plot.png")
    plt.savefig(sir_path)
    plt.close()

    # Reported cases plot
    plt.figure(figsize=(10, 4))
    plt.plot(df["Day"], df["Reported_Cases"], label="Reported Cases", color="purple")
    plt.xlabel("Day")
    plt.ylabel("Cases Reported")
    plt.title("Simulated Reported Cases (Noisy Observations)")
    plt.legend()
    plt.tight_layout()
    cases_path = os.path.join(outdir, f"{prefix}_reported_cases.png")
    plt.savefig(cases_path)
    plt.close()

    print(f"Saved plots -> {sir_path}, {cases_path}")


def generate_multiple_datasets(n_variants=3, outdir="datasets"):
    """
    Generate multiple synthetic datasets with different seeds.
    """
    for i in range(n_variants):
        seed = 42 + i
        df = generate_epidemic_data(random_seed=seed)
        filename = os.path.join(outdir, f"synthetic_epidemic_{i+1}.csv")
        save_dataset(df, filename)
        save_plots(df, outdir="plots", prefix=f"synthetic_epidemic_{i+1}")


if __name__ == "__main__":
    # Single run
    data = generate_epidemic_data()
    save_dataset(data, "datasets/single_epidemic.csv")
    save_plots(data, outdir="plots", prefix="single_epidemic")

    # Generate multiple variants
    generate_multiple_datasets(n_variants=3)
    print("All datasets and plots generated.")
