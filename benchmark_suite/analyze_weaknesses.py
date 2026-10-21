import os
import json
import pandas as pd
import numpy as np

RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"
SPLITS_DIR = r"D:\dataset\smart london\processed_grid_data\2_model_ready_splits"

def analyze_all_models():
    # Load test ground truth with full features
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"), parse_dates=['tstp'])
    # Models have lookback = 48 (first 48 half-hours skipped)
    eval_df = test_df.iloc[48:].copy().reset_index(drop=True)
    
    model_preds_files = {
        'LightGBM': 'preds_lightgbm.csv',
        'TCN-QRNN (Nature 2025)': 'preds_tcn_qrnn.csv',
        'TimeMixer (ICLR 2024)': 'preds_timemixer.csv',
        'DLinear (AAAI 2023)': 'preds_dlinear.csv',
        'LTE Hybrid (IEEE TSG 2025)': 'preds_lte_hybrid.csv',
        'iTransformer (ICLR 2024)': 'preds_itransformer.csv',
        'EMSformer (IEEE TPWRS 2025)': 'preds_emsformer.csv',
        'PatchTST (ICLR 2023)': 'preds_patchtst.csv',
        'Novel MS-STLFNet (Proposed Hybrid)': 'preds_novel_ms_stlfnet.csv',
        'Novel 1: MSD-TCN': 'preds_novel_1_msd_tcn.csv',
        'Novel 2: D-PABNet': 'preds_novel_2_dpabnet.csv',
        'Novel 3: H-MoE-Boost': 'preds_novel_3_h_moe_boost.csv'
    }

    # Load predictions
    preds_dict = {}
    for name, fname in model_preds_files.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            df_p = pd.read_csv(fpath)
            # Ensure length matches eval_df
            if len(df_p) == len(eval_df):
                preds_dict[name] = df_p['pred_kWh'].values
            elif len(df_p) == len(test_df):
                # LightGBM was evaluated on all rows
                preds_dict[name] = df_p['pred_kWh'].values[48:]

    actual = eval_df['grid_load_kWh'].values
    eval_df['actual_kWh'] = actual

    # 1. Hour-of-Day Analysis (Peak 17:00-21:00 vs Base 02:00-06:00)
    eval_df['hour_int'] = eval_df['tstp'].dt.hour
    is_peak = eval_df['hour_int'].isin([17, 18, 19, 20, 21])
    is_base = eval_df['hour_int'].isin([2, 3, 4, 5, 6])
    is_weekend = eval_df['is_weekend'] == 1
    is_cold = eval_df['temperature'] < eval_df['temperature'].quantile(0.20)

    breakdown_results = {}
    for name, pred in preds_dict.items():
        abs_err = np.abs(actual - pred)
        breakdown_results[name] = {
            'Overall_MAE': round(float(np.mean(abs_err)), 2),
            'Peak_Hour_MAE': round(float(np.mean(abs_err[is_peak])), 2),
            'Base_Load_MAE': round(float(np.mean(abs_err[is_base])), 2),
            'Weekend_MAE': round(float(np.mean(abs_err[is_weekend])), 2),
            'Cold_Snap_MAE': round(float(np.mean(abs_err[is_cold])), 2),
            'Max_Error': round(float(np.max(abs_err)), 2)
        }

    df_breakdown = pd.DataFrame(breakdown_results).T
    df_breakdown.to_csv(os.path.join(RESULTS_DIR, "error_breakdown_by_regime.csv"))
    print("\n=== FAILURE & ERROR BREAKDOWN BY REGIME ===")
    print(df_breakdown.to_string())

    # Save summary json
    with open(os.path.join(RESULTS_DIR, "error_breakdown_by_regime.json"), "w") as f:
        json.dump(breakdown_results, f, indent=4)

if __name__ == "__main__":
    analyze_all_models()
