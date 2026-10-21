import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"
SPLITS_DIR = r"D:\dataset\smart london\processed_grid_data\2_model_ready_splits"
FINAL_RESULTS_DIR = r"D:\dataset\smart london\results"

os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

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

def generate_all_figures():
    # Load test split
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"), parse_dates=['tstp'])
    eval_df = test_df.iloc[48:].copy().reset_index(drop=True)
    
    # -------------------------------------------------------------------------
    # 1. Figure 1: 7-Day Peak Winter Load Forecasting Tracking (Dec 2013)
    # Using LATEST CHAMPION: STLF-SuperEnsemble
    # -------------------------------------------------------------------------
    preds_files = {
        'Actual Ground Truth': None,
        'Champion: STLF-SuperEnsemble (Ours)': 'preds_champion_super_ensemble.csv',
        'LightGBM': 'preds_lightgbm.csv',
        'TCN-QRNN (Nature 2025)': 'preds_tcn_qrnn.csv',
        'TimeMixer (ICLR 2024)': 'preds_timemixer.csv',
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

    # Select winter peak week (336 half-hours = 7 days)
    # Peak winter week in late December (Dec 23-30, 2013)
    start_idx = 2600
    end_idx = start_idx + 336
    time_window = eval_df['tstp'].iloc[start_idx:end_idx]

    plt.figure(figsize=(15, 6.5), dpi=300)
    plt.plot(time_window, preds_data['Actual Ground Truth'][start_idx:end_idx], 
             label='Actual Grid Demand (Ground Truth)', color='black', linewidth=2.6, alpha=0.9, zorder=5)
    plt.plot(time_window, preds_data['Champion: STLF-SuperEnsemble (Ours)'][start_idx:end_idx], 
             label='Champion: STLF-SuperEnsemble (Ours - MAE 21.64)', color='#dc2626', linewidth=2.2, zorder=6)
    plt.plot(time_window, preds_data['LightGBM'][start_idx:end_idx], 
             label='LightGBM (Baseline - MAE 25.02)', color='#2563eb', linestyle='--', linewidth=1.5, alpha=0.85)
    plt.plot(time_window, preds_data['TCN-QRNN (Nature 2025)'][start_idx:end_idx], 
             label='TCN-QRNN (Nature 2025 - MAE 35.38)', color='#16a34a', linestyle='-.', linewidth=1.4, alpha=0.8)
    plt.plot(time_window, preds_data['TimeMixer (ICLR 2024)'][start_idx:end_idx], 
             label='TimeMixer (ICLR 2024 - MAE 42.38)', color='#9333ea', linestyle=':', linewidth=1.3, alpha=0.75)
    plt.plot(time_window, preds_data['EMSformer (IEEE TPWRS 2025)'][start_idx:end_idx], 
             label='EMSformer (IEEE 2025 - MAE 84.49)', color='#ea580c', linestyle=':', linewidth=1.2, alpha=0.7)

    plt.title('7-Day Grid Load Tracking During Winter Peak Demand (London Smart Grid: Dec 23–30, 2013)', fontweight='bold', pad=12)
    plt.xlabel('Timestamp (30-Minute Intervals)', fontweight='bold')
    plt.ylabel('Aggregated Grid Load (kWh)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper right', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cbd5e1')
    plt.tight_layout()
    
    fig1_dest1 = os.path.join(RESULTS_DIR, "fig1_load_tracking_peak_week.png")
    fig1_dest2 = os.path.join(FINAL_RESULTS_DIR, "Figure_1_load_tracking_peak_week.png")
    plt.savefig(fig1_dest1, dpi=300)
    plt.savefig(fig1_dest2, dpi=300)
    plt.close()
    print("Saved Figure 1 with Champion STLF-SuperEnsemble")

    # -------------------------------------------------------------------------
    # 2. Figure 2: Operational Regime Failure Breakdown
    # Using LATEST CHAMPION: STLF-SuperEnsemble as the leading architecture
    # -------------------------------------------------------------------------
    df_regime = pd.read_csv(os.path.join(RESULTS_DIR, "error_breakdown_by_regime.csv"), index_col=0)
    
    key_models = [
        'Champion: STLF-SuperEnsemble',
        'LightGBM', 
        'TCN-QRNN (Nature 2025)', 
        'TimeMixer (ICLR 2024)', 
        'DLinear (AAAI 2023)', 
        'LTE Hybrid (IEEE TSG 2025)', 
        'EMSformer (IEEE TPWRS 2025)'
    ]
    sub_df = df_regime.loc[key_models][['Overall_MAE', 'Peak_Hour_MAE', 'Base_Load_MAE', 'Cold_Snap_MAE']]
    
    colors = ['#10b981', '#3b82f6', '#f59e0b', '#8b5cf6']
    ax = sub_df.plot(kind='bar', figsize=(14, 6.5), color=colors, edgecolor='black', linewidth=0.8, width=0.8)
    plt.title('Operational Grid Regime Stress Comparison (Champion vs. Literature Models)', fontweight='bold', pad=12)
    plt.ylabel('Mean Absolute Error (kWh) - Lower is Better', fontweight='bold')
    plt.xlabel('Forecasting Model Architecture', fontweight='bold')
    plt.xticks(rotation=22, ha='right', fontweight='medium')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(['Overall Test MAE', 'Peak Hours (17:00-21:00)', 'Base Load (02:00-06:00)', 'Cold Snaps (<20th pct Temp)'],
               frameon=True, framealpha=0.95)
    
    # Highlight Champion bar group
    ax.patches[0].set_edgecolor('red')
    ax.patches[0].set_linewidth(2)
    
    plt.tight_layout()
    fig2_dest1 = os.path.join(RESULTS_DIR, "fig2_regime_stress_comparison.png")
    fig2_dest2 = os.path.join(FINAL_RESULTS_DIR, "Figure_2_regime_stress_comparison.png")
    plt.savefig(fig2_dest1, dpi=300)
    plt.savefig(fig2_dest2, dpi=300)
    plt.close()
    print("Saved Figure 2 with Champion STLF-SuperEnsemble")

    # -------------------------------------------------------------------------
    # 3. Figure 3: Pareto Frontier: Accuracy (MAE) vs Compute Time
    # Highlighting Champion STLF-SuperEnsemble as the Pareto Optimal Frontier
    # -------------------------------------------------------------------------
    df_lead = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_leaderboard.csv"))
    
    plt.figure(figsize=(11, 6.5), dpi=300)
    for _, row in df_lead.iterrows():
        name = row['model']
        mae = row['MAE_kWh']
        t = row['train_time_sec']
        
        if 'Champion' in name or 'SuperEnsemble' in name:
            color = '#dc2626'
            marker = '*'
            size = 320
            label = 'Champion (Ours)'
        elif 'LightGBM' in name:
            color = '#2563eb'
            marker = 's'
            size = 140
            label = 'GBDT'
        elif 'Nature' in name or 'QRNN' in name:
            color = '#16a34a'
            marker = 'o'
            size = 130
            label = 'Nature 2025'
        elif 'PatchTST' in name:
            color = '#94a3b8'
            marker = 'x'
            size = 110
            label = 'ICLR 2023'
        elif 'Novel' in name:
            color = '#f97316'
            marker = '^'
            size = 130
            label = 'Novel Ablations'
        else:
            color = '#8b5cf6'
            marker = 'D'
            size = 110
            label = 'Literature Baselines'

        plt.scatter(t, mae, color=color, marker=marker, s=size, edgecolors='black', linewidth=1.2, zorder=4)
        
        # Display clean labels
        disp_name = name.split(' (')[0].replace('Champion: ', '[Champion] ')
        offset_y = 2.5 if 'Champion' not in name else -3.5
        offset_x = 1.15 if 'PatchTST' not in name else 0.5
        plt.annotate(disp_name, (t * offset_x, mae + offset_y), fontsize=9, fontweight='bold' if 'Champion' in name else 'normal')

    # Draw Pareto boundary line connecting LightGBM and Champion
    plt.plot([0.28, 1.6], [21.64, 25.02], color='#dc2626', linestyle='--', linewidth=1.8, alpha=0.7, zorder=2)
    plt.annotate('Pareto Optimal Frontier', (0.45, 23.5), color='#dc2626', fontsize=9.5, fontweight='bold', rotation=-14)

    plt.xscale('log')
    plt.title('Accuracy vs. Computational Efficiency (Pareto Frontier)', fontweight='bold', pad=12)
    plt.xlabel('Training / Optimization Latency (Seconds, Log Scale)', fontweight='bold')
    plt.ylabel('Test MAE (kWh) - Lower is Better', fontweight='bold')
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    fig3_dest1 = os.path.join(RESULTS_DIR, "fig3_pareto_accuracy_vs_efficiency.png")
    fig3_dest2 = os.path.join(FINAL_RESULTS_DIR, "Figure_3_pareto_accuracy_vs_efficiency.png")
    plt.savefig(fig3_dest1, dpi=300)
    plt.savefig(fig3_dest2, dpi=300)
    plt.close()
    print("Saved Figure 3 with Champion STLF-SuperEnsemble")

    # -------------------------------------------------------------------------
    # 4. Figure 4: Boxplot of Absolute Residual Error Distribution
    # -------------------------------------------------------------------------
    models_box = {
        'Champion (Ours)': 'preds_champion_super_ensemble.csv',
        'LightGBM': 'preds_lightgbm.csv',
        'Novel 1 (MSD-TCN)': 'preds_novel_1_msd_tcn.csv',
        'Novel 2 (D-PABNet)': 'preds_novel_2_dpabnet.csv',
        'Novel 3 (H-MoE)': 'preds_novel_3_h_moe_boost.csv',
        'TCN-QRNN (2025)': 'preds_tcn_qrnn.csv',
        'TimeMixer (2024)': 'preds_timemixer.csv',
        'DLinear (2023)': 'preds_dlinear.csv',
        'LTE Hybrid (2025)': 'preds_lte_hybrid.csv',
        'iTransformer (2024)': 'preds_itransformer.csv',
        'EMSformer (2025)': 'preds_emsformer.csv'
    }

    error_data = []
    for name, fpath in models_box.items():
        df_p = pd.read_csv(os.path.join(RESULTS_DIR, fpath))
        errs = (df_p['actual_kWh'] - df_p['pred_kWh']).abs()
        for e in errs:
            error_data.append({'Model': name, 'Absolute Error (kWh)': e})

    err_df = pd.DataFrame(error_data)

    plt.figure(figsize=(14, 6.5), dpi=300)
    palette = ['#10B981' if 'Champion' in m else ('#3B82F6' if 'Novel' in m else '#6B7280') for m in models_box.keys()]
    ax = sns.boxplot(x='Model', y='Absolute Error (kWh)', data=err_df, palette=palette, showfliers=False, width=0.6)
    plt.title('Absolute Residual Error Distribution Across Models on London Test Set (Lower & Tighter is Better)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Forecasting Architecture', fontsize=12, fontweight='bold')
    plt.ylabel('Absolute Error (kWh)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=22, ha='right', fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    medians = err_df.groupby('Model')['Absolute Error (kWh)'].median().loc[list(models_box.keys())]
    for i, med in enumerate(medians):
        ax.text(i, med + 1.2, f'{med:.1f}', horizontalalignment='center', size=9.5, weight='bold', color='black')

    plt.tight_layout()
    fig4_dest = os.path.join(FINAL_RESULTS_DIR, "Figure_4_error_residuals_distribution.png")
    plt.savefig(fig4_dest, dpi=300)
    plt.close()
    print("Saved Figure 4 with Champion STLF-SuperEnsemble")

if __name__ == "__main__":
    generate_all_figures()
