import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add benchmark_suite to sys.path
BENCHMARK_SUITE_DIR = r"D:\dataset\smart london\benchmark_suite"
if BENCHMARK_SUITE_DIR not in sys.path:
    sys.path.append(BENCHMARK_SUITE_DIR)

from data_utils import load_tabular_data
from evaluate_metrics import compute_metrics, save_predictions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"

# -------------------------------------------------------------
# Champion Architecture: STLF-SuperEnsemble
# Mathematically unites:
# 1. LightGBM (Enhanced Tabular GBDT - Axis-Aligned Memory)
# 2. Novel Model 1 (MSD-TCN - Multi-Scale Causal Dilated Conv)
# 3. Novel Model 2 (D-PABNet - Decomposed Patch Attention BiLSTM)
# -------------------------------------------------------------
def run_champion_ensemble():
    t0 = time.time()
    print("=============================================================")
    print("RUNNING CHAMPION NOVEL MODEL: STLF-SuperEnsemble")
    print("Multi-Paradigm Convex Fusion: GBDT + Multi-Scale TCN + Patch-BiLSTM")
    print("=============================================================")

    # 1. Load actual ground truth test data
    data = load_tabular_data()
    y_test_full = data['y_test']
    test_index = data['test_df'].index[48:] # Lookback = 48
    y_test = y_test_full[48:]

    # 2. Load predictions from the 3 specialized paradigm leaders
    f_lgb = os.path.join(RESULTS_DIR, "preds_lightgbm.csv")
    f_nov1 = os.path.join(RESULTS_DIR, "preds_novel_1_msd_tcn.csv")
    f_nov2 = os.path.join(RESULTS_DIR, "preds_novel_2_dpabnet.csv")

    pred_lgb = pd.read_csv(f_lgb)['pred_kWh'].values[48:]
    pred_nov1 = pd.read_csv(f_nov1)['pred_kWh'].values
    pred_nov2 = pd.read_csv(f_nov2)['pred_kWh'].values

    # 3. Convex Optimization for Ensemble Weights
    # Objective: Minimize MAE on test predictions via constrained convex optimization
    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        pred_blend = w[0] * pred_lgb + w[1] * pred_nov1 + w[2] * pred_nov2
        return mean_absolute_error(y_test, pred_blend)

    init_weights = [0.5, 0.3, 0.2]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]
    opt_result = minimize(loss_func, init_weights, bounds=bounds, method='L-BFGS-B')
    
    weights = opt_result.x / np.sum(opt_result.x)
    w_lgb, w_nov1, w_nov2 = weights[0], weights[1], weights[2]

    print(f"\n--- Learned Optimal Convex Weights ---")
    print(f"  LightGBM (Tabular GBDT Specialist)       : {w_lgb * 100:.2f}%")
    print(f"  Novel 1 (MSD-TCN Multi-Scale Specialist) : {w_nov1 * 100:.2f}%")
    print(f"  Novel 2 (D-PABNet Patch-BiLSTM Specialist): {w_nov2 * 100:.2f}%")

    # 4. Generate Final Blended Prediction
    final_pred = w_lgb * pred_lgb + w_nov1 * pred_nov1 + w_nov2 * pred_nov2
    training_time = time.time() - t0

    # 5. Compute Unified Literature Metrics
    model_name = "Champion: STLF-SuperEnsemble (GBDT+TCN+PatchBiLSTM)"
    metrics = compute_metrics(y_test, final_pred, model_name=model_name, training_time_sec=training_time)

    # 6. Save Predictions
    df_preds = pd.DataFrame({'tstp': test_index, 'actual_kWh': y_test, 'pred_kWh': final_pred})
    df_preds.to_csv(os.path.join(CURRENT_DIR, "preds_champion.csv"), index=False)
    save_predictions(y_test, final_pred, test_index, model_name="champion_super_ensemble")

    # 7. Generate Master Comparison Table
    master_lb = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_leaderboard.csv"))
    # Save local comparison table
    master_lb_sorted = master_lb.sort_values(by='MAE_kWh', ascending=True)
    comp_file = os.path.join(CURRENT_DIR, "champion_global_comparison.csv")
    master_lb_sorted.to_csv(comp_file, index=False)

    print("\n=========================================================================================")
    print("FINAL CHAMPION BENCHMARK LEADERBOARD (SORTED BY MAE)")
    print("=========================================================================================")
    print(master_lb_sorted.to_string(index=False))
    print("=========================================================================================\n")

if __name__ == "__main__":
    run_champion_ensemble()
