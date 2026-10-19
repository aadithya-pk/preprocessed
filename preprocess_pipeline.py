import os
import glob
import json
import time
import pandas as pd
import numpy as np

def run_preprocessing():
    start_total_time = time.time()
    data_dir = r"D:\dataset\smart london"
    blocks_dir = os.path.join(data_dir, "halfhourly_dataset")
    weather_file = os.path.join(data_dir, "weather_hourly_darksky.csv")
    output_dir = os.path.join(data_dir, "processed_grid_data")

    # Target directories
    dir_verification = os.path.join(output_dir, "1_manual_verification")
    dir_splits = os.path.join(output_dir, "2_model_ready_splits")
    dir_metadata = os.path.join(output_dir, "3_metadata")

    os.makedirs(dir_verification, exist_ok=True)
    os.makedirs(dir_splits, exist_ok=True)
    os.makedirs(dir_metadata, exist_ok=True)

    cache_file = os.path.join(output_dir, "cache_grid_30m.csv")

    # -------------------------------------------------------------
    # 1. LOAD OR STREAM AGGREGATED LOAD & HOUSEHOLDS
    # -------------------------------------------------------------
    if os.path.exists(cache_file):
        print(f"\n[Phase 1] Found cached grid series at {cache_file}, loading directly...")
        df_grid = pd.read_csv(cache_file, index_col='tstp', parse_dates=['tstp'])
    else:
        block_files = sorted(glob.glob(os.path.join(blocks_dir, "block_*.csv")))
        total_blocks = len(block_files)
        print(f"Found {total_blocks} block files in {blocks_dir}")

        if total_blocks == 0:
            raise FileNotFoundError(f"No block files found in {blocks_dir}")

        master_energy_series = None
        master_count_series = None

        print(f"\n[Phase 1] Streaming and accumulating {total_blocks} block files...")
        for idx, fpath in enumerate(block_files, start=1):
            t0 = time.time()
            df_b = pd.read_csv(fpath, usecols=['tstp', 'energy(kWh/hh)'], dtype=str)
            df_b['energy'] = pd.to_numeric(df_b['energy(kWh/hh)'].str.strip(), errors='coerce').astype('float32')
            df_b['tstp'] = pd.to_datetime(df_b['tstp'].str.slice(0, 16))

            valid_b = df_b.dropna(subset=['energy'])
            block_sum = valid_b.groupby('tstp')['energy'].sum()
            block_count = valid_b.groupby('tstp')['energy'].count().astype('int32')

            if master_energy_series is None:
                master_energy_series = block_sum
                master_count_series = block_count
            else:
                master_energy_series = master_energy_series.add(block_sum, fill_value=0)
                master_count_series = master_count_series.add(block_count, fill_value=0)

            elapsed = time.time() - t0
            if idx % 10 == 0 or idx == total_blocks:
                print(f"  Processed [{idx:3d}/{total_blocks}] blocks... (last block took {elapsed:.2f}s)")

        df_grid = pd.DataFrame({
            'grid_load_kWh': master_energy_series,
            'n_households': master_count_series
        }).sort_index()

        df_grid = df_grid.asfreq('30min')
        df_grid['grid_load_kWh'] = df_grid['grid_load_kWh'].interpolate(method='time', limit=4)
        df_grid['n_households'] = df_grid['n_households'].interpolate(method='time', limit=4).round().astype('Int64')
        df_grid = df_grid.dropna()

        df_grid.to_csv(cache_file, index_label='tstp')
        print(f"Saved updated cache to: {cache_file}")

    print(f"Standardized continuous series: {len(df_grid):,} timestamps from {df_grid.index.min()} to {df_grid.index.max()}")

    # -------------------------------------------------------------
    # 2. LOAD & RESAMPLE WEATHER DATA (Temp, ApparentTemp, Humidity, Pressure)
    # -------------------------------------------------------------
    print(f"\n[Phase 2] Loading and aligning weather data from {weather_file}...")
    df_weather = pd.read_csv(
        weather_file,
        usecols=['time', 'temperature', 'apparentTemperature', 'humidity', 'pressure']
    )
    df_weather['time'] = pd.to_datetime(df_weather['time']).dt.floor('min')
    df_weather = df_weather.set_index('time').sort_index()

    # Interpolate 1-hour weather to 30-minute intervals
    df_weather_30m = df_weather.resample('30min').interpolate(method='time')

    # Merge weather with grid load
    df_merged = df_grid.join(df_weather_30m, how='inner')
    print(f"Merged dataset shape: {df_merged.shape}")

    # -------------------------------------------------------------
    # 3. FEATURE ENGINEERING (State-of-the-Art Feature Set)
    # -------------------------------------------------------------
    print("\n[Phase 3] Engineering calendar, lag, cyclical, and holiday features...")

    # A. Target Normalization (Per-household stationary load)
    df_merged['energy_per_household_kWh'] = (df_merged['grid_load_kWh'] / df_merged['n_households']).astype('float32')

    # B. Calendar features
    df_merged['hour'] = df_merged.index.hour
    df_merged['dayofweek'] = df_merged.index.dayofweek
    df_merged['is_weekend'] = (df_merged['dayofweek'] >= 5).astype(int)
    df_merged['month'] = df_merged.index.month

    # C. UK Bank Holidays
    uk_bank_holidays = {
        '2011-12-26', '2011-12-27',
        '2012-01-02', '2012-04-06', '2012-04-09', '2012-05-07', '2012-06-04',
        '2012-06-05', '2012-08-27', '2012-12-25', '2012-12-26',
        '2013-01-01', '2013-03-29', '2013-04-01', '2013-05-06', '2013-05-27',
        '2013-08-26', '2013-12-25', '2013-12-26',
        '2014-01-01'
    }
    date_strings = df_merged.index.strftime('%Y-%m-%d')
    df_merged['is_holiday'] = date_strings.isin(uk_bank_holidays).astype(int)

    # D. Cyclical sine/cosine encodings
    df_merged['sin_hour'] = np.sin(2 * np.pi * df_merged['hour'] / 24).astype('float32')
    df_merged['cos_hour'] = np.cos(2 * np.pi * df_merged['hour'] / 24).astype('float32')
    df_merged['sin_month'] = np.sin(2 * np.pi * df_merged['month'] / 12).astype('float32')
    df_merged['cos_month'] = np.cos(2 * np.pi * df_merged['month'] / 12).astype('float32')

    # E. Autoregressive Lags
    target = 'grid_load_kWh'
    df_merged['lag_1'] = df_merged[target].shift(1)       # 30 min ago
    df_merged['lag_2'] = df_merged[target].shift(2)       # 1 hour ago
    df_merged['lag_48'] = df_merged[target].shift(48)     # 24 hours ago (same time yesterday)
    df_merged['lag_336'] = df_merged[target].shift(336)   # 7 days ago (same time last week)

    # F. 24h Rolling Mean
    df_merged['rolling_mean_48'] = df_merged[target].shift(1).rolling(48).mean()

    # Drop initial NaN rows created by lag shifting (first 7 days warm-up)
    df_final = df_merged.dropna().copy()
    print(f"Final feature dataset shape: {df_final.shape}")

    # Reorder columns logically
    col_order = [
        'grid_load_kWh',
        'energy_per_household_kWh',
        'n_households',
        'temperature',
        'apparentTemperature',
        'humidity',
        'pressure',
        'hour',
        'dayofweek',
        'is_weekend',
        'is_holiday',
        'month',
        'sin_hour',
        'cos_hour',
        'sin_month',
        'cos_month',
        'lag_1',
        'lag_2',
        'lag_48',
        'lag_336',
        'rolling_mean_48'
    ]
    df_final = df_final[col_order]

    # -------------------------------------------------------------
    # 4. CHRONOLOGICAL SPLITTING & REPLACEMENT
    # -------------------------------------------------------------
    print("\n[Phase 4] Replacing all files inside processed_grid_data...")

    # Remove any stray secondary verification files
    alt_file = os.path.join(dir_verification, "verification_sample_200_rows_updated.csv")
    if os.path.exists(alt_file):
        try:
            os.remove(alt_file)
        except Exception:
            pass

    # A. Chronological Time-Series Splits (70% Train, 15% Val, 15% Test)
    total_len = len(df_final)
    idx_train_end = int(total_len * 0.70)
    idx_val_end = int(total_len * 0.85)

    train_df = df_final.iloc[:idx_train_end]
    val_df = df_final.iloc[idx_train_end:idx_val_end]
    test_df = df_final.iloc[idx_val_end:]

    train_path = os.path.join(dir_splits, "train.csv")
    val_path = os.path.join(dir_splits, "val.csv")
    test_path = os.path.join(dir_splits, "test.csv")

    train_df.to_csv(train_path, index_label='tstp')
    val_df.to_csv(val_path, index_label='tstp')
    test_df.to_csv(test_path, index_label='tstp')

    print(f"  [+] Replaced Train split: {train_path} ({len(train_df):,} rows)")
    print(f"  [+] Replaced Val split:   {val_path} ({len(val_df):,} rows)")
    print(f"  [+] Replaced Test split:  {test_path} ({len(test_df):,} rows)")

    # B. 200 rows for manual verification
    sample_file = os.path.join(dir_verification, "verification_sample_200_rows.csv")
    df_final.head(200).to_csv(sample_file, index_label='tstp')
    print(f"  [+] Replaced Verification Sample: {sample_file}")

    # C. Metadata Summary
    summary = {
        "dataset_name": "Smart Meters in London - Enhanced Aggregated Grid Load",
        "total_cleaned_samples": total_len,
        "splits": {
            "train": {
                "rows": len(train_df),
                "date_range": [str(train_df.index.min()), str(train_df.index.max())]
            },
            "validation": {
                "rows": len(val_df),
                "date_range": [str(val_df.index.min()), str(val_df.index.max())]
            },
            "test": {
                "rows": len(test_df),
                "date_range": [str(test_df.index.min()), str(test_df.index.max())]
            }
        },
        "target_features": ["grid_load_kWh", "energy_per_household_kWh"],
        "input_features": [c for c in df_final.columns if c not in ["grid_load_kWh", "energy_per_household_kWh"]],
        "all_features": list(df_final.columns),
        "null_counts": df_final.isnull().sum().to_dict(),
        "total_runtime_seconds": round(time.time() - start_total_time, 2)
    }

    summary_file = os.path.join(dir_metadata, "dataset_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"  [+] Replaced Summary Metadata: {summary_file}")

    print(f"\n=======================================================")
    print(f"PREPROCESSING COMPLETED IN {summary['total_runtime_seconds']} SECONDS!")
    print(f"=======================================================\n")

if __name__ == "__main__":
    run_preprocessing()
