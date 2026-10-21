import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"
SPLITS_DIR = r"D:\dataset\smart london\processed_grid_data\2_model_ready_splits"

def compute_full_union_evaluation():
    # Load ground truth test set
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"), parse_dates=['tstp'])
    eval_df = test_df.iloc[48:].copy().reset_index(drop=True)
    y_true = eval_df['grid_load_kWh'].values
    total_consumption = float(np.sum(y_true))
    mean_true = float(np.mean(y_true))

    # All 12 models: 8 Base Models + 4 Novel Models
    models_dict = {
        # --- Base Literature Models ---
        'LightGBM': {'file': 'preds_lightgbm.csv', 'time': 1.60, 'source': 'Tabular Baseline'},
        'TCN-QRNN': {'file': 'preds_tcn_qrnn.csv', 'time': 447.28, 'source': 'Nature Sci. Rep. 2025'},
        'TimeMixer': {'file': 'preds_timemixer.csv', 'time': 28.49, 'source': 'ICLR 2024'},
        'DLinear': {'file': 'preds_dlinear.csv', 'time': 13.57, 'source': 'AAAI 2023'},
        'LTE Hybrid': {'file': 'preds_lte_hybrid.csv', 'time': 5672.18, 'source': 'IEEE TSG May 2025'},
        'iTransformer': {'file': 'preds_itransformer.csv', 'time': 217.37, 'source': 'ICLR 2024'},
        'EMSformer': {'file': 'preds_emsformer.csv', 'time': 836.04, 'source': 'IEEE TPWRS Nov 2025'},
        'PatchTST': {'file': 'preds_patchtst.csv', 'time': 719.77, 'source': 'ICLR 2023'},
        # --- Novel Models (Ours) ---
        'Novel 1: MSD-TCN': {'file': 'preds_novel_1_msd_tcn.csv', 'time': 434.36, 'source': 'Novel Hybrid 1 (Ours)'},
        'Novel 2: D-PABNet': {'file': 'preds_novel_2_dpabnet.csv', 'time': 188.63, 'source': 'Novel Hybrid 2 (Ours)'},
        'Novel 3: H-MoE-Boost': {'file': 'preds_novel_3_h_moe_boost.csv', 'time': 198.53, 'source': 'Novel Hybrid 3 (Ours)'},
        'Champion: STLF-SuperEnsemble': {'file': 'preds_champion_super_ensemble.csv', 'time': 0.28, 'source': 'Novel Model 4 Champion'}
    }

    # Operational Stress Regimes
    eval_df['hour_int'] = eval_df['tstp'].dt.hour
    is_peak = eval_df['hour_int'].isin([17, 18, 19, 20, 21]).values
    is_base = eval_df['hour_int'].isin([2, 3, 4, 5, 6]).values
    is_cold = (eval_df['temperature'] < eval_df['temperature'].quantile(0.20)).values
    is_weekend = (eval_df['is_weekend'] == 1).values

    full_results = []

    for name, info in models_dict.items():
        fpath = os.path.join(RESULTS_DIR, info['file'])
        if not os.path.exists(fpath):
            continue
        df_p = pd.read_csv(fpath)
        if len(df_p) == len(eval_df):
            y_pred = df_p['pred_kWh'].values
        elif len(df_p) == len(test_df):
            y_pred = df_p['pred_kWh'].values[48:]
        else:
            continue

        errors = y_pred - y_true
        abs_errors = np.abs(errors)

        # 1. Absolute Errors
        mae = float(mean_absolute_error(y_true, y_pred))
        mse = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(mse))

        # 2. Relative & Percentage Errors
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - mean_true) ** 2))
        rse = float(ss_res / (ss_tot + 1e-9)) # Relative Squared Error (IEEE TPWRS 2025)
        r2 = float(r2_score(y_true, y_pred))
        wape = float(np.sum(abs_errors) / total_consumption * 100.0) # WAPE (%)
        mape = float(np.mean(abs_errors / np.maximum(y_true, 1e-5)) * 100.0) # MAPE (%)
        pmae = float((mae / total_consumption) * 1e5) # PMAE x 10^-5 (IEEE TSG 2025)

        # 3. Robustness & Dispersion
        err_std = float(np.std(errors))
        max_err = float(np.max(abs_errors))

        # 4. Operational Regimes
        peak_mae = float(np.mean(abs_errors[is_peak]))
        base_mae = float(np.mean(abs_errors[is_base]))
        cold_mae = float(np.mean(abs_errors[is_cold]))
        weekend_mae = float(np.mean(abs_errors[is_weekend]))

        full_results.append({
            'Model': name,
            'Source': info['source'],
            'MAE_kWh': round(mae, 2),
            'RMSE_kWh': round(rmse, 2),
            'MSE': round(mse, 2),
            'RSE': round(rse, 4),
            'R2': round(r2, 4),
            'WAPE_pct': round(wape, 2),
            'MAPE_pct': round(mape, 2),
            'PMAE_x1e5': round(pmae, 3),
            'Err_Std_kWh': round(err_std, 2),
            'Max_Err_kWh': round(max_err, 2),
            'Peak_MAE_kWh': round(peak_mae, 2),
            'Base_MAE_kWh': round(base_mae, 2),
            'Cold_MAE_kWh': round(cold_mae, 2),
            'Weekend_MAE_kWh': round(weekend_mae, 2),
            'Time_s': round(info['time'], 2)
        })

    res_df = pd.DataFrame(full_results).sort_values(by='MAE_kWh', ascending=True)
    out_csv = os.path.join(RESULTS_DIR, "comprehensive_union_metrics.csv")
    res_df.to_csv(out_csv, index=False)
    print("Saved comprehensive union evaluation to:", out_csv)
    print("\n" + res_df.to_string(index=False))

if __name__ == "__main__":
    compute_full_union_evaluation()
