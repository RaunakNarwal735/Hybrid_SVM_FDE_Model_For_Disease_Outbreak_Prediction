"""
Epidemic Dataset Generator

Two modes:
    1) Random auto mode (default): python epidemic_generator.py
    2) Interactive mode:          python epidemic_generator.py interact

Generates a stochastic SIR-style epidemic with:
- Mask & crowdedness behavior modifying beta
- Quarantine reducing infectious pool
- Seasonality (optional)
- Intervention (optional; fixed window)
- Multi-wave (optional; late beta boost)
- Central-peaked epidemic envelope (Gaussian)
- Random dropper: irregular case drops (simulate reporting/behavior shocks)
- Reporting probability ramp + day-to-day jitter
- Outputs dataset.csv, params.json, and plots in timestamped run folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys
import random
from datetime import datetime

# =========================================================
# GLOBAL CONFIG
# =========================================================
BASE_SAVE_DIR = r"C:\Users\rishu narwal\Desktop\SVM_FDE\datasets"

# Behavioral beta scaling caps
MASK_MAX_REDUCTION = 0.6       # mask_score=10 -> -60% beta
CROWD_MAX_INCREASE = 0.7       # crowd_score=10 -> +70% beta

# Seasonality defaults
SEASONAL_AMP = 0.25            # +/- 25%
SEASONAL_PERIOD = 60           # 60-day cycle

# Quarantine
QUARANTINE_FRACTION = 0.5      # 50% of infected effectively isolated if enabled

# Recovery rate (gamma)
GAMMA = 0.08                   # slower recovery than before -> longer epidemic

# Central peak envelope (Gaussian)
CENTER_AMP = 0.5               # up to +50% beta at mid-epidemic
CENTER_SIGMA_FRAC = 0.2        # width = fraction of total days (std dev)

# Multi-wave second bump (multiplicative)
MULTIWAVE_BETA_MULT = 1.5      # 50% beta increase during second wave
MULTIWAVE_DURATION = 20        # days

# Intervention window (if enabled)
INT_DURATION = 14              # days
INT_MULTIPLIER = 0.4           # 60% beta reduction

# Reporting jitter
REPORT_SIGMA_ABS = 0.05        # absolute +/- noise to reporting prob
REPORT_CLIP_MIN = 0.0
REPORT_CLIP_MAX = 0.99

# Random dropper
DROP_MIN_INTERVAL = 15         # min days between drops
DROP_MAX_INTERVAL = 20         # max days between drops
DROP_MIN_CASES = 1
DROP_MAX_CASES = 100

# Matplotlib non-interactive backend safety (in case)
plt.switch_backend("Agg")


# =========================================================
# HELPERS
# =========================================================
def ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def yn_to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("y", "yes", "true", "1")


# =========================================================
# INTERACTIVE INPUTS
# =========================================================
def get_user_inputs():
    """
    Collect user inputs interactively with defaults and validation.
    Returns:
        dict of parameters (strings/ints/floats/flags)
    """
    print("\n=== INTERACTIVE MODE ===")

    # population
    pop = input("Population [73500]: ").strip()
    population = int(pop) if pop else 73500
    if population > 1_000_000:
        print("Population capped at 1,000,000.")
        population = 1_000_000
    if population < 1000:
        print("Population too low; forcing 1000.")
        population = 1000

    # days
    d = input("Number of days [180]: ").strip()
    days = int(d) if d else 180
    if days < 30:
        print("Days too low; forcing 30.")
        days = 30

    # initial infected
    ii = input("Initial infected [50]: ").strip()
    initial_infected = int(ii) if ii else 50
    max_init = max(1, min(population // 20, 5000))  # <=5% of pop, cap 5000
    if initial_infected > max_init:
        print(f"Initial infected capped at {max_init}.")
        initial_infected = max_init
    if initial_infected < 1:
        initial_infected = 1

    # mask and crowd
    ms = input("Mask adherence 1-10 [4]: ").strip()
    mask_score = int(ms) if ms else 4
    mask_score = min(max(mask_score, 1), 10)

    cs = input("Crowdedness 1-10 [6]: ").strip()
    crowdedness_score = int(cs) if cs else 6
    crowdedness_score = min(max(crowdedness_score, 1), 10)

    # feature flags
    quarantine_enabled = input("Enable quarantine (y/n) [y]: ").strip() or "y"
    seasonality_enabled = input("Enable seasonality (y/n) [y]: ").strip() or "y"
    interventions_enabled = input("Enable interventions (y/n) [y]: ").strip() or "y"
    multi_wave = input("Enable multi-wave (y/n) [n]: ").strip() or "n"

    # reporting range
    rpmin = input("Reporting probability min [0.6]: ").strip()
    reporting_prob_min = float(rpmin) if rpmin else 0.6
    rpmax = input("Reporting probability max [0.85]: ").strip()
    reporting_prob_max = float(rpmax) if rpmax else 0.85
    if reporting_prob_max <= reporting_prob_min:
        print("Max must be > min; adjusting.")
        reporting_prob_max = min(0.95, reporting_prob_min + 0.1)

    # seed
    seed_in = input("Random seed [auto]: ").strip()
    random_seed = None if not seed_in or seed_in.lower() == "auto" else int(seed_in)

    params = {
        "population": population,
        "days": days,
        "initial_infected": initial_infected,
        "mask_score": mask_score,
        "crowdedness_score": crowdedness_score,
        "quarantine_enabled": quarantine_enabled,
        "seasonality_enabled": seasonality_enabled,
        "interventions_enabled": interventions_enabled,
        "reporting_prob_min": reporting_prob_min,
        "reporting_prob_max": reporting_prob_max,
        "multi_wave": multi_wave,
        "random_seed": random_seed,
        "mode": "interactive"
    }
    return params


# =========================================================
# RANDOM PARAM GENERATOR
# =========================================================
def generate_random_inputs():
    """
    Generate a random but realistic parameter config.
    Returns:
        dict of parameters
    """
    population = random.randint(10_000, 1_000_000)
    days = random.randint(90, 365)
    max_init = max(1, min(population // 20, 5000))
    initial_infected = random.randint(10, max_init)

    mask_score = random.randint(1, 10)
    crowdedness_score = random.randint(1, 10)

    quarantine_enabled = random.choice(["y", "n"])
    seasonality_enabled = random.choice(["y", "n"])
    interventions_enabled = random.choice(["y", "n"])
    multi_wave = random.choice(["y", "n"])

    low = round(random.uniform(0.3, 0.6), 2)
    high = round(random.uniform(low + 0.1, 0.95), 2)

    random_seed = random.randint(1, 999999)

    params = {
        "population": population,
        "days": days,
        "initial_infected": initial_infected,
        "mask_score": mask_score,
        "crowdedness_score": crowdedness_score,
        "quarantine_enabled": quarantine_enabled,
        "seasonality_enabled": seasonality_enabled,
        "interventions_enabled": interventions_enabled,
        "reporting_prob_min": low,
        "reporting_prob_max": high,
        "multi_wave": multi_wave,
        "random_seed": random_seed,
        "mode": "random"
    }
    return params


# =========================================================
# SIMULATOR
# =========================================================
def simulate_epidemic(params):
    """
    Run stochastic epidemic simulation using modified SIR dynamics.

    Beta pipeline:
        beta_base -> behavior (mask, crowd) -> seasonality -> intervention -> multi-wave -> jitter -> center envelope -> sustain rule

    Quarantine reduces the effective infectious pool.

    Returns:
        pd.DataFrame with agreed column schema.
    """
    # RNG
    seed = params["random_seed"]
    rng = np.random.default_rng(seed)

    # Unpack
    N = params["population"]
    days = params["days"]
    S = N - params["initial_infected"]
    I = params["initial_infected"]
    R = 0

    # Base beta
    beta_base = 0.3

    # Behavior adjustment
    mask_factor = params["mask_score"] / 10.0
    crowd_factor = params["crowdedness_score"] / 10.0
    beta_behaviour = beta_base * (1 - MASK_MAX_REDUCTION * mask_factor)
    beta_behaviour *= (1 + CROWD_MAX_INCREASE * crowd_factor)

    # Flags
    seasonality = yn_to_bool(params["seasonality_enabled"])
    quarantine = yn_to_bool(params["quarantine_enabled"])
    interventions = yn_to_bool(params["interventions_enabled"])
    multi_wave = yn_to_bool(params["multi_wave"])

    # Reporting schedule + jitter
    p_min, p_max = params["reporting_prob_min"], params["reporting_prob_max"]
    base_report_probs = np.linspace(p_min, p_max, days)

    # Intervention window(auto)
    if interventions:
        int_start = days // 3
        int_end = int_start + INT_DURATION
    else:
        int_start = int_end = -1  # never triggers

    # Multi-wave bump
    if multi_wave:
        wave_start = days // 2
        wave_end = wave_start + MULTIWAVE_DURATION
    else:
        wave_start = wave_end = -1

    # Central Gaussian envelope to push peak toward middle
    center_mu = days / 2
    center_sigma = days * CENTER_SIGMA_FRAC  # convert fraction to days

    # Random dropper scheduler
    next_drop_day = rng.integers(DROP_MIN_INTERVAL, DROP_MAX_INTERVAL + 1)

    # Storage
    rows = []
    for t in range(days):
        # --- Beta build-up ---
        beta_t = beta_behaviour

        # Seasonality
        if seasonality:
            season_index = 1 + SEASONAL_AMP * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
            beta_t *= season_index
        else:
            season_index = 1.0

        # Intervention
        if int_start <= t <= int_end:
            beta_t *= INT_MULTIPLIER
            int_flag = 1
        else:
            int_flag = 0

        # Multiwave
        if wave_start <= t <= wave_end:
            beta_t *= MULTIWAVE_BETA_MULT
            wave_flag = 1
        else:
            wave_flag = 0

        # Daily jitter around 0 mean (mild, 5% stdev proportional)
        beta_t *= (1 + rng.normal(0, 0.05))

        # Center peak envelope (Gaussian)
        # (1 + CENTER_AMP*exp(...)) ensures mid-curve boosting
        center_scale = 1 + CENTER_AMP * np.exp(-0.5 * ((t - center_mu) / center_sigma) ** 2)
        beta_t *= center_scale

        # Sustain if epidemic collapses too early
        # If infections fall below very low value after first third, gently boost
        if I < max(5, 0.001 * N) and t > days // 3:
            beta_t *= 1.2

        # Quarantine effect
        if quarantine:
            eff_I = I * (1 - QUARANTINE_FRACTION)
            q_frac = QUARANTINE_FRACTION
        else:
            eff_I = I
            q_frac = 0.0

        # Transmission & recovery probabilities
        p_inf = 1.0 - np.exp(-beta_t * eff_I / N)
        p_inf = np.clip(p_inf, 0, 1)
        p_rec = 1.0 - np.exp(-GAMMA)
        p_rec = np.clip(p_rec, 0, 1)

        # New transitions (stochastic)
        new_inf = rng.binomial(S, p_inf) if S > 0 else 0
        new_rec = rng.binomial(I, p_rec) if I > 0 else 0

        # Apply compartment updates
        S -= new_inf
        I += new_inf - new_rec
        R += new_rec

        # Random dropper event?
        if t == next_drop_day and I > 0:
            drop_cases = rng.integers(DROP_MIN_CASES, min(DROP_MAX_CASES, I) + 1)
            I -= drop_cases
            R += drop_cases
            # schedule next drop
            next_drop_day += rng.integers(DROP_MIN_INTERVAL, DROP_MAX_INTERVAL + 1)

        # Reporting jitter
        base_p = base_report_probs[t]
        report_p = base_p + rng.normal(0, REPORT_SIGMA_ABS)
        report_p = float(np.clip(report_p, REPORT_CLIP_MIN, REPORT_CLIP_MAX))

        # Observed cases (under-reporting)
        reported_cases = rng.binomial(new_inf, report_p) if new_inf > 0 else 0

        # Store
        rows.append([
            t, S, I, R, new_inf, new_rec, reported_cases,
            beta_t, season_index, int_flag, q_frac, report_p, wave_flag
        ])

    df = pd.DataFrame(rows, columns=[
        "Day", "Susceptible", "Infected", "Recovered",
        "New_Infections", "New_Recoveries", "Reported_Cases",
        "Beta_Effective", "Season_Index", "Intervention_Flag",
        "Quarantine_Fraction", "Reporting_Prob", "MultiWave_Flag"
    ])
    return df


# =========================================================
# PLOTTING
# =========================================================
def plot_sir(df, path):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(10, 6))
    plt.stackplot(df["Day"],  df["Infected"], df["Susceptible"],df["Recovered"],
                  labels=[ "Infected","Susceptible", "Recovered"],
                  colors=[ "#ff6961","#77b5fe", "#77dd77"])
    plt.legend(loc="upper right")
    plt.title("Epidemic Simulation (SIR Compartments)")
    plt.xlabel("Day")
    plt.ylabel("Population")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reported(df, path):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(10, 5))
    plt.plot(df["Day"], df["Reported_Cases"], label="Reported Cases", color="orange")
    plt.title("Reported Cases Over Time")
    plt.xlabel("Day")
    plt.ylabel("Cases")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =========================================================
# OUTPUTS
# =========================================================
def save_outputs(df, params, out_dir):
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "dataset.csv")
    json_path = os.path.join(out_dir, "params.json")
    sir_plot = os.path.join(out_dir, "stacked_sir.png")
    reported_plot = os.path.join(out_dir, "reported_cases.png")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(params, f, indent=4)

    plot_sir(df, sir_plot)
    plot_reported(df, reported_plot)

    print(f"\nData and plots saved in: {out_dir}")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")
    print(f"  Plots: {sir_plot}, {reported_plot}")


# =========================================================
# MAIN
# =========================================================
def main():
    # Mode selection
    if len(sys.argv) > 1 and sys.argv[1].lower() == "interact":
        params = get_user_inputs()
    else:
        params = generate_random_inputs()

    # Show params
    print("\n=== PARAMETERS ===")
    print(json.dumps(params, indent=4))

    # Run simulation
    df = simulate_epidemic(params)

    # Save
    run_id = ts()
    out_dir = os.path.join(BASE_SAVE_DIR, f"run_{run_id}")
    save_outputs(df, params, out_dir)

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
