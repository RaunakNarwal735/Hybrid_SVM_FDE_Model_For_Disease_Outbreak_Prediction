#!/usr/bin/env python
"""
real_to_BIGBOY.py

Convert WHO-style COVID-19 data into BIGBOY format for a specific country and date range.

Expected columns in CSV:
    Column1 (date), Country_code, Country, WHO_region, New_cases,
    Cumulative_cases, New_deaths, Cumulative_deaths
"""

import argparse
import os
import pandas as pd
import numpy as np
from datetime import datetime

# ==================== UTILS ==================== #
def parse_args():
    parser = argparse.ArgumentParser(description="Convert COVID-19 data to BIGBOY format.")
    parser.add_argument("--csv", required=True, help="Path to COVID dataset CSV.")
    parser.add_argument("--country", required=True, help="Country name (case-insensitive).")
    parser.add_argument("--start_date", required=True, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end_date", required=True, help="End date (YYYY-MM-DD).")
    parser.add_argument("--population", type=int, required=True, help="Population of the country.")
    parser.add_argument("--infectious_days", type=int, default=10, help="Rolling days for active cases.")
    parser.add_argument("--reporting_prob", type=float, default=0.5, help="Assumed reporting probability for beta calc.")
    parser.add_argument("--season_mode", choices=["india", "sin"], default="india", help="Seasonality mode.")
    parser.add_argument("--out_dir", default="datasets", help="Where to save output dataset.")
    return parser.parse_args()

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def build_season_index(dates, mode="india"):
    """
    Build a detailed seasonality factor based on Indian weather + festival cycles.
    """
    if mode == "sin":
        # Pure sinusoidal
        day_of_year = dates.dt.dayofyear
        return 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.25)

    # Month-based baseline
    month = dates.dt.month
    base_index = np.where(month.isin([12, 1, 2]), 1.15,     # Winter
                  np.where(month.isin([3, 4, 5]), 1.05,     # Pre-monsoon summer
                  np.where(month.isin([6, 7, 8, 9]), 0.90,  # Monsoon
                  np.where(month.isin([10, 11]), 1.20, 1.0))))

    # Festival/event spikes
    spike = np.zeros(len(dates))
    for i, d in enumerate(dates):
        if (d.month == 10 and 20 <= d.day <= 30):  # Diwali window
            spike[i] = 0.1
        if (d.month == 3 and 5 <= d.day <= 15):    # Holi window
            spike[i] = 0.08

    season_index = base_index + spike

    # Smooth transitions (rolling mean)
    season_index = pd.Series(season_index).rolling(7, min_periods=1, center=True).mean().values
    return season_index


# ==================== CORE LOGIC ==================== #
def convert_real_data(args):
    df = pd.read_csv(args.csv)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Auto-rename
    rename_map = {
        'column1': 'date',
        'date_reported': 'date',
        'country': 'country',
        'new_cases': 'new_cases',
        'cumulative_cases': 'cumulative_cases',
        'new_deaths': 'new_deaths',
        'cumulative_deaths': 'cumulative_deaths'
    }
    df = df.rename(columns=rename_map)

    # Check required columns
    for col in ['date', 'country', 'new_cases', 'cumulative_cases', 'new_deaths', 'cumulative_deaths']:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # Filter country
    df = df[df['country'].str.lower() == args.country.lower()]
    if df.empty:
        raise ValueError(f"No data found for country: {args.country}")

    # Parse date and filter
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    start, end = pd.to_datetime(args.start_date), pd.to_datetime(args.end_date)
    df = df[(df['date'] >= start) & (df['date'] <= end)]

    if df.empty:
        raise ValueError(f"No data in range {args.start_date} - {args.end_date} for {args.country}")

    df = df.sort_values(by='date').reset_index(drop=True)
    for col in ['new_cases', 'cumulative_cases', 'new_deaths', 'cumulative_deaths']:
        df[col] = df[col].fillna(0).clip(lower=0)

    # Construct BIGBOY columns
    df['Day'] = np.arange(len(df))
    df['Reported_Cases'] = df['new_cases'].astype(int)

    # Infected = rolling sum of new cases over infectious_days
    active = df['new_cases'].rolling(args.infectious_days, min_periods=1).sum()
    active = active - df['new_deaths'].rolling(7, min_periods=1).sum().fillna(0)
    df['Infected'] = active.clip(lower=0).astype(int)

    # Recovered = cumulative_cases - active - deaths
    df['Recovered'] = (df['cumulative_cases'] - df['Infected'] - df['cumulative_deaths']).clip(lower=0).astype(int)

    # Susceptible = population - cumulative_cases
    df['Susceptible'] = (args.population - df['cumulative_cases']).clip(lower=0).astype(int)

    # Beta_Effective calculation
    S_prev = df['Susceptible'].shift(1).replace(0, np.nan)
    I_prev = df['Infected'].shift(1).replace(0, np.nan)
    new_inf = df['new_cases'] / max(args.reporting_prob, 1e-6)
    beta_eff = (new_inf * args.population) / (S_prev * I_prev)
    df['Beta_Effective'] = beta_eff.replace([np.inf, -np.inf], np.nan).rolling(7, min_periods=1).mean().fillna(0.0)

    # Seasonality
    df['Season_Index'] = build_season_index(df['date'], mode=args.season_mode)

    bigboy_df = df[['Day', 'Susceptible', 'Infected', 'Recovered', 'Reported_Cases', 'Beta_Effective', 'Season_Index']]

    return bigboy_df, df

# ==================== MAIN ==================== #
def main():
    args = parse_args()
    ensure_dir(args.out_dir)

    bigboy_df, raw_df = convert_real_data(args)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_folder = os.path.join(args.out_dir, f"{args.country}_{ts}")
    ensure_dir(out_folder)

    out_csv = os.path.join(out_folder, "dataset.csv")
    bigboy_df.to_csv(out_csv, index=False)

    print(f"[INFO] BIGBOY dataset saved: {out_csv}")
    print(f"[INFO] Rows: {len(bigboy_df)} | Dates: {raw_df['date'].min().date()} to {raw_df['date'].max().date()}")

if __name__ == "__main__":
    main()
