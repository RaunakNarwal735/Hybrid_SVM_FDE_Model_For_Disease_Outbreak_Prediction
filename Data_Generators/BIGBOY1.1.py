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
import argparse

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

# Add new global config for vaccination, variant, and mask decay
default_vaccination_rate = 0.0  # default: 0% per day
VARIANT_BETA_MULTIPLIER_DEFAULT = 1.5
VARIANT_DAY_DEFAULT = 60
INCUBATION_PERIOD_DEFAULT = 4  # days
MASK_DECAY_RATE_DEFAULT = 0.01  # 1% per day

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

    # vaccination
    vaccination_enabled = input("Enable vaccination (y/n) [n]: ").strip() or "n"
    daily_vaccination_rate = input("Daily vaccination rate (fraction, e.g. 0.01 for 1%) [0.0]: ").strip()
    daily_vaccination_rate = float(daily_vaccination_rate) if daily_vaccination_rate else default_vaccination_rate

    # incubation period
    incubation_period = input(f"Incubation period (days) [{INCUBATION_PERIOD_DEFAULT}]: ").strip()
    incubation_period = int(incubation_period) if incubation_period else INCUBATION_PERIOD_DEFAULT

    # variant
    variant_enabled = input("Enable variant emergence (y/n) [n]: ").strip() or "n"
    variant_day = input(f"Variant emergence day [{VARIANT_DAY_DEFAULT}]: ").strip()
    variant_day = int(variant_day) if variant_day else VARIANT_DAY_DEFAULT
    variant_beta_multiplier = input(f"Variant beta multiplier [{VARIANT_BETA_MULTIPLIER_DEFAULT}]: ").strip()
    variant_beta_multiplier = float(variant_beta_multiplier) if variant_beta_multiplier else VARIANT_BETA_MULTIPLIER_DEFAULT

    # testing intensity
    testing_rate = input("Testing rate (low/medium/high) [medium]: ").strip() or "medium"

    # mask decay
    mask_decay_rate = input(f"Mask decay rate per day (fraction, e.g. 0.01 for 1%) [{MASK_DECAY_RATE_DEFAULT}]: ").strip()
    mask_decay_rate = float(mask_decay_rate) if mask_decay_rate else MASK_DECAY_RATE_DEFAULT

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
        "vaccination_enabled": vaccination_enabled,
        "daily_vaccination_rate": daily_vaccination_rate,
        "incubation_period": incubation_period,
        "variant_enabled": variant_enabled,
        "variant_day": variant_day,
        "variant_beta_multiplier": variant_beta_multiplier,
        "testing_rate": testing_rate,
        "mask_decay_rate": mask_decay_rate,
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

    vaccination_enabled = random.choice(["y", "n"])
    daily_vaccination_rate = round(random.uniform(0.0, 0.02), 3)  # up to 2% per day
    incubation_period = random.randint(2, 7)
    variant_enabled = random.choice(["y", "n"])
    variant_day = random.randint(30, 120)
    variant_beta_multiplier = round(random.uniform(1.2, 2.0), 2)
    testing_rate = random.choice(["low", "medium", "high"])
    mask_decay_rate = round(random.uniform(0.005, 0.02), 4)

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
        "vaccination_enabled": vaccination_enabled,
        "daily_vaccination_rate": daily_vaccination_rate,
        "incubation_period": incubation_period,
        "variant_enabled": variant_enabled,
        "variant_day": variant_day,
        "variant_beta_multiplier": variant_beta_multiplier,
        "testing_rate": testing_rate,
        "mask_decay_rate": mask_decay_rate,
        "mode": "random"
    }
    return params


# =========================================================
# SIMULATOR
# =========================================================
def simulate_epidemic(params, use_sir=False):
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

    # Unpack new params
    vaccination = yn_to_bool(params.get("vaccination_enabled", "n"))
    daily_vaccination_rate = params.get("daily_vaccination_rate", 0.0)
    incubation_period = params.get("incubation_period", INCUBATION_PERIOD_DEFAULT)
    variant = yn_to_bool(params.get("variant_enabled", "n"))
    variant_day = params.get("variant_day", VARIANT_DAY_DEFAULT)
    variant_beta_multiplier = params.get("variant_beta_multiplier", VARIANT_BETA_MULTIPLIER_DEFAULT)
    testing_rate = params.get("testing_rate", "medium")
    mask_decay_rate = params.get("mask_decay_rate", MASK_DECAY_RATE_DEFAULT)

    # SEIR compartments
    E = 0
    I = params["initial_infected"]
    R = 0
    E_queue = [0] * incubation_period  # queue for exposed individuals

    # Storage
    rows = []
    for t in range(days):
        # --- Beta build-up ---
        # Mask decay over time
        mask_factor = max(0, params["mask_score"] / 10.0 - mask_decay_rate * t)
        crowd_factor = params["crowdedness_score"] / 10.0
        beta_behaviour = beta_base * (1 - MASK_MAX_REDUCTION * mask_factor)
        beta_behaviour *= (1 + CROWD_MAX_INCREASE * crowd_factor)

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

        # Variant emergence
        if variant and t >= variant_day:
            beta_t *= variant_beta_multiplier

        # Vaccination
        if vaccination and S > 0:
            vaccinated_today = min(S, int(S * daily_vaccination_rate))
            S -= vaccinated_today
            R += vaccinated_today

        # Transmission & recovery probabilities
        p_inf = 1.0 - np.exp(-beta_t * eff_I / N)
        p_inf = np.clip(p_inf, 0, 1)
        p_rec = 1.0 - np.exp(-GAMMA)
        p_rec = np.clip(p_rec, 0, 1)

        # New transitions (stochastic)
        new_exp = rng.binomial(S, p_inf) if S > 0 else 0
        S -= new_exp
        E_queue.append(new_exp)
        new_inf = E_queue.pop(0)
        E += new_exp - new_inf
        new_rec = rng.binomial(I, p_rec) if I > 0 else 0
        I += new_inf - new_rec
        R += new_rec

        # Random dropper event?
        if t == next_drop_day and I > 0:
            drop_cases = rng.integers(DROP_MIN_CASES, min(DROP_MAX_CASES, I) + 1)
            I -= drop_cases
            R += drop_cases
            # schedule next drop
            next_drop_day += rng.integers(DROP_MIN_INTERVAL, DROP_MAX_INTERVAL + 1)

        # Testing intensity affects reporting probability and lag
        if testing_rate == "low":
            report_p = max(base_report_probs[t] * 0.7, 0.01)
            report_lag = 2
        elif testing_rate == "high":
            report_p = min(base_report_probs[t] * 1.2, 0.99)
            report_lag = 0
        else:
            report_p = base_report_probs[t]
            report_lag = 1
        # Reporting lag: shift reported cases by lag days
        if t >= report_lag:
            reported_cases = rng.binomial(new_inf, report_p) if new_inf > 0 else 0
        else:
            reported_cases = 0

        # Effective reproduction number Rt
        Rt = (beta_t / GAMMA) * (S / N) if N > 0 else 0

        # Store
        rows.append([
            t, S, E, I, R, new_exp, new_inf, new_rec, reported_cases,
            beta_t, season_index, int_flag, q_frac, report_p, wave_flag, Rt
        ])

    df = pd.DataFrame(rows, columns=[
        "Day", "Susceptible", "Exposed", "Infected", "Recovered",
        "New_Exposed", "New_Infections", "New_Recoveries", "Reported_Cases",
        "Beta_Effective", "Season_Index", "Intervention_Flag",
        "Quarantine_Fraction", "Reporting_Prob", "MultiWave_Flag", "Rt"
    ])
    return df


# =========================================================
# PLOTTING
# =========================================================
def plot_sir(df, path):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(10, 6))
    plt.stackplot(df["Day"],  df["Infected"], df["Susceptible"], df["Exposed"], df["Recovered"],
                  labels=["Infected", "Susceptible", "Exposed", "Recovered"],
                  colors=["#ff6961", "#77b5fe", "#fdfd96", "#77dd77"])
    plt.legend(loc="upper right")
    plt.title("Epidemic Simulation (SEIR Compartments)")
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
def save_outputs(df, params, out_dir, save_plots=True):
    ensure_dir(out_dir)
    csv_path = os.path.join(out_dir, "dataset.csv")
    json_path = os.path.join(out_dir, "params.json")
    sir_plot = os.path.join(out_dir, "stacked_sir.png")
    reported_plot = os.path.join(out_dir, "reported_cases.png")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(params, f, indent=4)

    if save_plots:
        plot_sir(df, sir_plot)
        plot_reported(df, reported_plot)
        print(f"  Plots: {sir_plot}, {reported_plot}")

    print(f"\nData and plots saved in: {out_dir}")
    print(f"  CSV  -> {csv_path}")
    print(f"  JSON -> {json_path}")


# =========================================================
# MAIN
# =========================================================
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('mode', nargs='?', default=None)
    parser.add_argument('count', nargs='?', type=int, default=5)
    parser.add_argument('--sir', action='store_true', help='Use SIR model instead of SEIR')
    parser.add_argument('--plots', dest='save_plots', action='store_true', help='Save plots (default)')
    parser.add_argument('--no-plots', dest='save_plots', action='store_false', help='Do not save plots')
    parser.add_argument('--help', action='store_true', help='Show help message and exit')
    parser.set_defaults(save_plots=True)
    args, unknown = parser.parse_known_args()

    if args.help:
        print_help()
        return

    # Mode selection
    if args.mode == 'interact':
        params = get_user_inputs()
        df = simulate_epidemic(params, use_sir=args.sir)
        out_dir = os.path.join(BASE_SAVE_DIR, f"run_{ts()}")
        save_outputs(df, params, out_dir, save_plots=args.save_plots)
    elif args.mode == 'batch':
        count = args.count
        generate_batch(count, use_sir=args.sir, save_plots=args.save_plots)
    elif args.mode is None:
        params = generate_random_inputs()
        df = simulate_epidemic(params, use_sir=args.sir)
        out_dir = os.path.join(BASE_SAVE_DIR, f"run_{ts()}")
        save_outputs(df, params, out_dir, save_plots=args.save_plots)
    else:
        print("Unknown mode! Use: 'interact', 'batch <count>', or --help.")


def print_help():
    print("""
Epidemic Dataset Generator CLI Usage:

python BIGBOY1.1.py [mode] [options]

Modes:
  interact           Interactive mode (prompts for parameters)
  batch [N]          Generate a batch of N random datasets (default N=5)

Options:
  --sir              Use SIR model instead of SEIR (default: SEIR)
  --plots            Save plots (default: enabled)
  --no-plots         Do not save plots
  --help             Show this help message and exit

Examples:
  python BIGBOY1.1.py interact
  python BIGBOY1.1.py batch 10 --sir --no-plots
  python BIGBOY1.1.py --help
""")


def generate_batch(count=5, use_sir=False, save_plots=True):
    batch_id = ts()
    base_batch_dir = os.path.join(BASE_SAVE_DIR, f"batch_{batch_id}")
    ensure_dir(base_batch_dir)

    print(f"\n=== BATCH GENERATION: {count} datasets ===")
    for i in range(1, count + 1):
        params = generate_random_inputs()
        df = simulate_epidemic(params, use_sir=use_sir)
        run_dir = os.path.join(base_batch_dir, f"run_{i:03d}")
        ensure_dir(run_dir)
        csv_path = os.path.join(run_dir, "dataset.csv")
        json_path = os.path.join(run_dir, "params.json")
        df.to_csv(csv_path, index=False)
        with open(json_path, "w") as f:
            json.dump(params, f, indent=4)
        if save_plots:
            sir_plot = os.path.join(run_dir, "stacked_sir.png")
            reported_plot = os.path.join(run_dir, "reported_cases.png")
            plot_sir(df, sir_plot)
            plot_reported(df, reported_plot)
        print(f"  Dataset {i}/{count} -> {csv_path}")
    print(f"\nAll datasets saved in: {base_batch_dir}")

if __name__ == "__main__":
    main()
