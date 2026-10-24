import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = r"D:\dataset\smart london\results"
ALL_COMPARE_DIR = r"D:\dataset\smart london\all_compare_results"
BENCH_DIR = r"D:\dataset\smart london\benchmark_results"

os.makedirs(ALL_COMPARE_DIR, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 300
})

def build_all_comparison_suite():
    print("Building all_compare_results suite...")

    # Copy the established core figures
    shutil.copy(os.path.join(RESULTS_DIR, "Figure_1_load_tracking_peak_week.png"),
                os.path.join(ALL_COMPARE_DIR, "Fig1_load_tracking_winter_peak_week.png"))
    shutil.copy(os.path.join(RESULTS_DIR, "Figure_2_regime_stress_comparison.png"),
                os.path.join(ALL_COMPARE_DIR, "Fig2_regime_stress_operational_breakdown.png"))
    shutil.copy(os.path.join(RESULTS_DIR, "Figure_3_pareto_accuracy_vs_efficiency.png"),
                os.path.join(ALL_COMPARE_DIR, "Fig3_pareto_accuracy_vs_computational_latency.png"))
    shutil.copy(os.path.join(RESULTS_DIR, "Figure_4_error_residuals_distribution.png"),
                os.path.join(ALL_COMPARE_DIR, "Fig4_residual_error_distribution_boxplot.png"))
    shutil.copy(os.path.join(RESULTS_DIR, "Figure_5_ensemble_weights_and_correlation.png"),
                os.path.join(ALL_COMPARE_DIR, "Fig7_ensemble_weights_and_error_correlation.png"))
    print("Copied Fig 1, 2, 3, 4, 7.")

    # Load aligned predictions
    models = {
        'Champion (Ours)': 'preds_champion_super_ensemble.csv',
        'LightGBM': 'preds_lightgbm.csv',
        'TCN-QRNN (Nature 2025)': 'preds_tcn_qrnn.csv',
        'TimeMixer (ICLR 2024)': 'preds_timemixer.csv',
        'DLinear (AAAI 2023)': 'preds_dlinear.csv',
        'LTE Hybrid (IEEE TSG 2025)': 'preds_lte_hybrid.csv',
        'EMSformer (IEEE 2025)': 'preds_emsformer.csv'
    }

    dfs = {}
    for name, fpath in models.items():
        df = pd.read_csv(os.path.join(BENCH_DIR, fpath))
        df['tstp'] = pd.to_datetime(df['tstp'])
        dfs[name] = df.set_index('tstp')

    base_idx = dfs['Champion (Ours)'].index
    actual = dfs['Champion (Ours)']['actual_kWh'].values
    aligned_preds = {}
    for name, df in dfs.items():
        aligned_preds[name] = df.loc[base_idx]['pred_kWh'].values

    # -------------------------------------------------------------
    # Fig 5: Empirical Cumulative Distribution Function (ECDF)
    # (Standard in IEEE TPWRS & Nature forecasting papers)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6), dpi=300)
    colors = {
        'Champion (Ours)': '#dc2626',
        'LightGBM': '#2563eb',
        'TCN-QRNN (Nature 2025)': '#16a34a',
        'TimeMixer (ICLR 2024)': '#9333ea',
        'DLinear (AAAI 2023)': '#64748b',
        'LTE Hybrid (IEEE TSG 2025)': '#0284c7',
        'EMSformer (IEEE 2025)': '#ea580c'
    }

    for name, pred in aligned_preds.items():
        err = np.abs(actual - pred)
        sorted_err = np.sort(err)
        ecdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        lw = 2.6 if 'Champion' in name else 1.6
        plt.plot(sorted_err, ecdf, label=name, color=colors[name], linewidth=lw)

    plt.axhline(0.8, color='#64748b', linestyle='--', alpha=0.7)
    plt.text(8, 0.82, '80% Confidence Bound', color='#475569', fontsize=9.5, fontweight='bold')
    plt.xlim(0, 160)
    plt.ylim(0, 1.02)
    plt.title('Empirical Cumulative Distribution Function (ECDF) of Forecast Errors\n(Higher curve = Better accuracy across all test intervals)',
              fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Absolute Error Bound |y - ŷ| (kWh)', fontsize=11, fontweight='bold')
    plt.ylabel('Cumulative Probability P(Error ≤ X)', fontsize=11, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower right', frameon=True, framealpha=0.95)
    plt.tight_layout()
    fig5_path = os.path.join(ALL_COMPARE_DIR, "Fig5_empirical_cumulative_distribution_ecdf.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print("Generated Fig 5 ECDF.")

    # -------------------------------------------------------------
    # Fig 6: Parity Scatter Plot (Actual vs Predicted Load)
    # (Standard in Nature, Applied Energy & IEEE papers)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=300)
    
    selected_evals = [
        ('Champion: STLF-SuperEnsemble (Ours)', aligned_preds['Champion (Ours)'], '#dc2626', 0.9950, 21.64),
        ('LightGBM (Top Literature Baseline)', aligned_preds['LightGBM'], '#2563eb', 0.9934, 25.02),
        ('TCN-QRNN (Nature Sci. Rep. 2025)', aligned_preds['TCN-QRNN (Nature 2025)'], '#16a34a', 0.9858, 35.38),
        ('EMSformer (IEEE TPWRS Nov 2025)', aligned_preds['EMSformer (IEEE 2025)'], '#ea580c', 0.9331, 84.49)
    ]

    min_val = min(actual) - 50
    max_val = max(actual) + 50

    for ax, (title, pred, col, r2, mae) in zip(axes, selected_evals):
        ax.scatter(actual, pred, color=col, alpha=0.25, s=6, edgecolors='none', rasterized=True)
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='Ideal Fit (y = x)')
        ax.set_title(f"{title}\nR² = {r2:.4f} | MAE = {mae:.2f} kWh", fontsize=10.5, fontweight='bold')
        ax.set_xlabel('Actual Grid Load (kWh)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Predicted Grid Load (kWh)', fontsize=10, fontweight='bold')
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='upper left', frameon=True, fontsize=9)

    plt.suptitle('Actual vs. Predicted Parity Alignment Across 5,861 Test Intervals\n(Points closer to the 45° diagonal line indicate superior prediction fidelity)',
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    fig6_path = os.path.join(ALL_COMPARE_DIR, "Fig6_parity_scatter_actual_vs_predicted.png")
    plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Fig 6 Parity Scatter.")

    # Also copy Tables into all_compare_results
    shutil.copy(os.path.join(RESULTS_DIR, "Table_1_benchmark_leaderboard.csv"),
                os.path.join(ALL_COMPARE_DIR, "Table_1_benchmark_leaderboard.csv"))
    shutil.copy(os.path.join(RESULTS_DIR, "Table_2_regime_stress_test.csv"),
                os.path.join(ALL_COMPARE_DIR, "Table_2_regime_stress_test.csv"))
    shutil.copy(os.path.join(RESULTS_DIR, "Table_3_statistical_significance_dm_test.csv"),
                os.path.join(ALL_COMPARE_DIR, "Table_3_statistical_significance_dm_test.csv"))
    print("Copied Tables 1, 2, 3 into all_compare_results.")

if __name__ == "__main__":
    build_all_comparison_suite()
