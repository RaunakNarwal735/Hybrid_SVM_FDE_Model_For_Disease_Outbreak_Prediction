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
DROP_MIN_INTERVAL = 1         # min days between drops
DROP_MAX_INTERVAL = 3       # max days between drops
DROP_MIN_CASES = 1
DROP_MAX_CASES = 17

# Add new global config for vaccination, variant, and mask decay
default_vaccination_rate = 0.0  # default: 0% per day
VARIANT_BETA_MULTIPLIER_DEFAULT = 1.5
VARIANT_DAY_DEFAULT = 60
INCUBATION_PERIOD_DEFAULT = 4  # days
MASK_DECAY_RATE_DEFAULT = 0.01  # 1% per day

# Multi-wave second bump (multiplicative)
MULTIWAVE_BETA_MULT = 1.5      # 50% beta increase during second wave
MULTIWAVE_DURATION = 20        # days

# Intervention window (if enabled)
INT_DURATION = 14              # days
INT_MULTIPLIER = 0.4           # 60% beta reduction

# Variant wave duration
VARIANT_WAVE_DURATION = 40  # days for a more dramatic, realistic wave

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
    daily_vaccination_rate = float(daily_vaccination_rate) if daily_vaccination_rate else 0.0

    # incubation period
    incubation_period = input(f"Incubation period (days) [{INCUBATION_PERIOD_DEFAULT}]: ").strip()
    incubation_period = int(incubation_period) if incubation_period else INCUBATION_PERIOD_DEFAULT

    # Unified variant/multi-wave
    num_waves = input("How many variant/multi-wave events? [1]: ").strip()
    num_waves = int(num_waves) if num_waves else 1
    waves = []
    for i in range(num_waves):
        print(f"--- Wave {i+1} ---")
        wave_day = input(f"  Day of wave [e.g., {60 + i*30}]: ").strip()
        wave_day = int(wave_day) if wave_day else (60 + i*30)
        wave_beta = input("  Beta multiplier [e.g., 2.5 for dramatic wave]: ").strip()
        wave_beta = float(wave_beta) if wave_beta else 2.5  # More dramatic default
        wave_seed = input("  Seed new exposed (number) [e.g., 100 for dramatic wave]: ").strip()
        wave_seed = int(wave_seed) if wave_seed else 100  # More dramatic default
        waves.append({"day": wave_day, "beta": wave_beta, "seed": wave_seed})

    # testing intensity
    testing_rate = input("Testing rate (low/medium/high) [medium]: ").strip() or "medium"

    # mask decay
    mask_decay_rate = input(f"Mask decay rate per day (fraction, e.g. 0.01 for 1%) [{MASK_DECAY_RATE_DEFAULT}]: ").strip()
    mask_decay_rate = float(mask_decay_rate) if mask_decay_rate else MASK_DECAY_RATE_DEFAULT

    # travel between cities
    travel_enabled = input("Enable travel between cities (y/n) [n]: ").strip() or "n"
    travel_max = 0
    if yn_to_bool(travel_enabled):
        travel_max_in = input("Travel (number of new exposed imported per day, 1–10) [3]: ").strip()
        travel_max = int(travel_max_in) if travel_max_in else 3

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
        "waves": waves,
        "testing_rate": testing_rate,
        "mask_decay_rate": mask_decay_rate,
        "travel_enabled": travel_enabled,
        "travel_max": travel_max,
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
    testing_rate = random.choice(["low", "medium", "high"])
    mask_decay_rate = round(random.uniform(0.005, 0.02), 4)

    travel_enabled = random.choice(["y", "n"])
    travel_max = random.randint(1, 10) if yn_to_bool(travel_enabled) else 0

    low = round(random.uniform(0.3, 0.6), 2)
    high = round(random.uniform(low + 0.1, 0.95), 2)

    random_seed = random.randint(1, 999999)

    # Unified variant/multi-wave
    num_waves = random.randint(1, 2)
    waves = []
    for i in range(num_waves):
        wave_day = random.randint(30 + i*30, 120 + i*30)
        wave_beta = round(random.uniform(1.2, 2.5), 2)
        wave_seed = random.randint(20, 100)
        waves.append({"day": wave_day, "beta": wave_beta, "seed": wave_seed})

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
        "waves": waves,
        "testing_rate": testing_rate,
        "mask_decay_rate": mask_decay_rate,
        "travel_enabled": travel_enabled,
        "travel_max": travel_max,
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
    testing_rate = params.get("testing_rate", "medium")
    mask_decay_rate = params.get("mask_decay_rate", MASK_DECAY_RATE_DEFAULT)
    travel = yn_to_bool(params.get("travel_enabled", "n"))
    travel_max = params.get("travel_max", 0)
    waves = params.get("waves", [])
    mask_score = params.get("mask_score", 4)
    crowdedness_score = params.get("crowdedness_score", 6)

    # SEIR compartments
    E = 0
    I = params["initial_infected"]
    R = 0
    E_queue = [0] * incubation_period  # queue for exposed individuals

    # Storage
    rows = []
    beta_rw = 0.0  # random walk component for beta
    gamma_rw = 0.0  # random walk component for gamma
    for t in range(days):
        # --- Realistic beta build-up ---
        # Mask effect (0–50% reduction)
        mask_effect = 1 - 0.05 * mask_score
        # Crowding effect (0–45% increase)
        crowd_effect = 1 + 0.05 * (crowdedness_score - 1)
        # Base beta
        beta_t = beta_base * mask_effect * crowd_effect
        # Seasonality
        if seasonality:
            beta_t *= 1 + 0.2 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
        # Intervention
        if int_start <= t <= int_end:
            beta_t *= 0.5
        # Multi-wave/variant
        beta_multiplier = 1.0
        for wave in waves:
            if wave["day"] <= t < wave["day"] + VARIANT_WAVE_DURATION:
                beta_multiplier *= wave["beta"]
        beta_t *= beta_multiplier
        # Daily jitter (10%)
        beta_t *= (1 + rng.normal(0, 0.10))
        # Add slow random walk to beta (stronger)
        beta_rw += rng.normal(0, 0.02)
        beta_rw = np.clip(beta_rw, -0.3, 0.3)
        beta_t *= (1 + beta_rw)

        # Quarantine effect (50% reduction)
        if quarantine:
            eff_I = I * 0.5
            q_frac = 0.5
        else:
            eff_I = I
            q_frac = 0.0

        # Minimum infection floor: if I is very low and S is available, seed a few new infections
        if I < 5 and S > 10:
            new_seed = min(20, S)
            S -= new_seed
            E += new_seed

        # Vaccination (90% efficacy)
        if vaccination and S > 0:
            vaccinated_today = min(S, int(S * daily_vaccination_rate))
            effective_vaccinated = int(vaccinated_today * 0.9)
            S -= effective_vaccinated
            R += effective_vaccinated
            # 10% remain susceptible (breakthroughs)

        # Travel importation: add travel_max new exposed from outside
        if travel and travel_max > 0:
            new_travelers = travel_max
            E += new_travelers
            E_queue[-1] += new_travelers

        # Inject new exposed for each wave on its start day
        for wave in waves:
            if t == wave["day"] and S > 0:
                new_wave_cases = min(wave["seed"], S)
                S -= new_wave_cases
                E += new_wave_cases
                E_queue[-1] += new_wave_cases

        # Apply all active waves (variants): boost beta for duration
        beta_multiplier = 1.0
        for wave in waves:
            if wave["day"] <= t < wave["day"] + VARIANT_WAVE_DURATION:
                beta_multiplier *= wave["beta"]
        beta_t *= beta_multiplier

        # Transmission & recovery probabilities
        p_inf = 1.0 - np.exp(-beta_t * eff_I / N)
        p_inf = np.clip(p_inf, 0, 1)

        # Gamma (recovery rate) with random walk and daily jitter (stronger)
        gamma_t = GAMMA * (1 + gamma_rw + rng.normal(0, 0.10))  # 10% daily jitter
        gamma_rw += rng.normal(0, 0.01)
        gamma_rw = np.clip(gamma_rw, -0.15, 0.15)
        p_rec = 1.0 - np.exp(-gamma_t)
        p_rec = np.clip(p_rec, 0, 1)

        # Robust minimum infected floor: every 3 days, if both I and E are zero and S > 10, seed 15–30 new infections (split E/I)
        if t > 0 and t % 3 == 0 and I + E == 0 and S > 10:
            new_seed = min(rng.integers(15, 31), S)
            S -= new_seed
            e_seed = new_seed // 2
            i_seed = new_seed - e_seed
            E += e_seed
            I += i_seed
            E_queue[-1] += e_seed

        # New transitions (stochastic)
        new_exp = rng.binomial(S, p_inf) if S > 0 else 0
        S -= new_exp
        E_queue.append(new_exp)
        new_inf = E_queue.pop(0)
        # E->I floor: always if E > 0
        if new_inf == 0 and E > 0:
            new_inf = 1
            E -= 1
        E += new_exp - new_inf
        new_rec = rng.binomial(I, p_rec) if I > 0 else 0
        I += new_inf - new_rec
        R += new_rec

        # Random dropper: at random intervals, add random number of new exposed
        if t == next_drop_day:
            new_cases = rng.integers(DROP_MIN_CASES, DROP_MAX_CASES + 1)
            E += new_cases
            E_queue[-1] += new_cases
            # Schedule next drop
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

        # Set output flags for DataFrame
        season_index = 1.0
        if seasonality:
            season_index = 1 + 0.2 * np.sin(2 * np.pi * t / SEASONAL_PERIOD)
        int_flag = 1 if int_start <= t <= int_end else 0
        wave_flag = 0
        for wave in waves:
            if wave["day"] <= t < wave["day"] + VARIANT_WAVE_DURATION:
                wave_flag = 1
                break

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
    plt.stackplot(df["Day"],  df["Infected"], df["Exposed"], df["Susceptible"],  df["Recovered"],
                  labels=["Infected","Exposed", "Susceptible",  "Recovered"],
                  colors=["#ff6961",  "#fdfd96","#77b5fe", "#77dd77"])
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
