import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"
SPLITS_DIR = r"D:\dataset\smart london\processed_grid_data\2_model_ready_splits"

plt.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300
})

def generate_figures():
    # Load test split
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"), parse_dates=['tstp'])
    eval_df = test_df.iloc[48:].copy().reset_index(drop=True)
    
    # Load predictions
    preds_files = {
        'Actual Ground Truth': None,
        'LightGBM': 'preds_lightgbm.csv',
        'TCN-QRNN (Nature 2025)': 'preds_tcn_qrnn.csv',
        'TimeMixer (ICLR 2024)': 'preds_timemixer.csv',
        'Novel MS-STLFNet (Ours)': 'preds_novel_ms_stlfnet.csv',
        'LTE Hybrid (IEEE TSG 2025)': 'preds_lte_hybrid.csv',
        'EMSformer (IEEE TPWRS 2025)': 'preds_emsformer.csv'
    }

    preds_data = {'Actual Ground Truth': eval_df['grid_load_kWh'].values}
    for name, fname in preds_files.items():
        if fname:
            df_p = pd.read_csv(os.path.join(RESULTS_DIR, fname))
            if len(df_p) == len(eval_df):
                preds_data[name] = df_p['pred_kWh'].values
            else:
                preds_data[name] = df_p['pred_kWh'].values[48:]

    # -------------------------------------------------------------
    # Figure 1: 7-Day Peak Winter Load Forecasting Tracking (Dec 2013)
    # -------------------------------------------------------------
    # Select winter peak week (336 half-hours = 7 days)
    # Peak winter occurs in December: index ~ 2500 to 2836
    start_idx = 2600
    end_idx = start_idx + 336
    time_window = eval_df['tstp'].iloc[start_idx:end_idx]

    plt.figure(figsize=(15, 6))
    plt.plot(time_window, preds_data['Actual Ground Truth'][start_idx:end_idx], 
             label='Actual Grid Demand', color='black', linewidth=2.5, alpha=0.9)
    plt.plot(time_window, preds_data['LightGBM'][start_idx:end_idx], 
             label='LightGBM', color='#1f77b4', linestyle='--', linewidth=1.5)
    plt.plot(time_window, preds_data['TCN-QRNN (Nature 2025)'][start_idx:end_idx], 
             label='TCN-QRNN (Nature 2025)', color='#2ca02c', linestyle='-.', linewidth=1.5)
    plt.plot(time_window, preds_data['Novel MS-STLFNet (Ours)'][start_idx:end_idx], 
             label='Novel MS-STLFNet (Ours)', color='#d62728', linewidth=1.8)
    plt.plot(time_window, preds_data['TimeMixer (ICLR 2024)'][start_idx:end_idx], 
             label='TimeMixer (ICLR 2024)', color='#9467bd', linestyle=':', linewidth=1.3)
    plt.plot(time_window, preds_data['EMSformer (IEEE TPWRS 2025)'][start_idx:end_idx], 
             label='EMSformer (IEEE 2025)', color='#ff7f0e', linestyle=':', linewidth=1.2, alpha=0.8)

    plt.title('7-Day Grid Load Forecasting Tracking During Winter Peak Demand (London Smart Grid)', fontweight='bold')
    plt.xlabel('Timestamp (30-Minute Intervals)')
    plt.ylabel('Aggregated Grid Load (kWh)')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', frameon=True, framealpha=0.9)
    plt.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, "fig1_load_tracking_peak_week.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # Figure 2: Operational Regime Failure Breakdown
    # -------------------------------------------------------------
    df_regime = pd.read_csv(os.path.join(RESULTS_DIR, "error_breakdown_by_regime.csv"), index_col=0)
    # Filter key models to display clearly
    key_models = [
        'LightGBM', 
        'TCN-QRNN (Nature 2025)', 
        'Novel MS-STLFNet (Proposed Hybrid)',
        'TimeMixer (ICLR 2024)', 
        'DLinear (AAAI 2023)', 
        'LTE Hybrid (IEEE TSG 2025)', 
        'EMSformer (IEEE TPWRS 2025)'
    ]
    sub_df = df_regime.loc[key_models][['Overall_MAE', 'Peak_Hour_MAE', 'Base_Load_MAE', 'Cold_Snap_MAE']]
    
    ax = sub_df.plot(kind='bar', figsize=(13, 6), colormap='viridis', edgecolor='black', width=0.8)
    plt.title('Model Robustness Across Critical Operational Stress Regimes', fontweight='bold')
    plt.ylabel('Mean Absolute Error (kWh)')
    plt.xlabel('Forecasting Architecture')
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(['Overall MAE', 'Peak Hours (17:00-21:00)', 'Base Load (02:00-06:00)', 'Cold Snaps (<20th pct)'])
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, "fig2_regime_stress_comparison.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # Figure 3: Pareto Frontier: Accuracy (MAE) vs Compute Time
    # -------------------------------------------------------------
    df_lead = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_leaderboard.csv"))
    
    plt.figure(figsize=(10, 6))
    for _, row in df_lead.iterrows():
        name = row['model']
        mae = row['MAE_kWh']
        t = row['train_time_sec']
        
        # Color highlight
        if 'Novel' in name:
            color = 'red'
            marker = '*'
            size = 200
        elif 'LightGBM' in name:
            color = 'blue'
            marker = 's'
            size = 120
        elif 'Nature' in name:
            color = 'green'
            marker = 'o'
            size = 120
        elif 'PatchTST' in name:
            color = 'gray'
            marker = 'x'
            size = 100
        else:
            color = 'purple'
            marker = '^'
            size = 110

        plt.scatter(t, mae, color=color, marker=marker, s=size, edgecolors='black', linewidth=1.2, zorder=3)
        plt.annotate(name, (t * 1.15, mae), fontsize=9, fontweight='medium')

    plt.xscale('log')
    plt.title('Accuracy vs. Computational Efficiency (Pareto Trade-off)', fontweight='bold')
    plt.xlabel('Training Time in Seconds (Log Scale)')
    plt.ylabel('Test MAE (kWh) - Lower is Better')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig3_path = os.path.join(RESULTS_DIR, "fig3_pareto_accuracy_vs_efficiency.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

if __name__ == "__main__":
    generate_figures()
