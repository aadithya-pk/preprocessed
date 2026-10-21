import pandas as pd
import numpy as np

# Load all predictions
models = {
    'Champion': 'preds_champion_super_ensemble.csv',
    'LightGBM': 'preds_lightgbm.csv',
    'DLinear': 'preds_dlinear.csv',
    'TCN_QRNN': 'preds_tcn_qrnn.csv',
    'LTE_Hybrid': 'preds_lte_hybrid.csv',
    'iTransformer': 'preds_itransformer.csv',
    'TimeMixer': 'preds_timemixer.csv',
    'PatchTST': 'preds_patchtst.csv',
    'EMSformer': 'preds_emsformer.csv',
    'MS_STLFNet': 'preds_novel_ms_stlfnet.csv'
}

dfs = {}
for name, fpath in models.items():
    df = pd.read_csv('benchmark_results/' + fpath)
    df['tstp'] = pd.to_datetime(df['tstp'])
    dfs[name] = df.set_index('tstp')

base = dfs['Champion'][['actual_kWh', 'pred_kWh']].rename(columns={'pred_kWh': 'pred_Champion'})
base['err_Champion'] = (base['actual_kWh'] - base['pred_Champion']).abs()

for name, df in dfs.items():
    if name != 'Champion':
        base['pred_' + name] = df['pred_kWh']
        base['err_' + name] = (base['actual_kWh'] - df['pred_kWh']).abs()

print("="*80)
print("AUDIT 1: WORST CASE / FAILURE MODE AUDIT FOR CHAMPION (TRANSPARENCY)")
print("="*80)
worst_champ = base.sort_values(by='err_Champion', ascending=False).head(5)
for idx, row in worst_champ.iterrows():
    print(f"Timestamp: {idx} | Actual: {row['actual_kWh']:7.2f} kWh | Champ Pred: {row['pred_Champion']:7.2f} kWh | Champ Error: {row['err_Champion']:6.2f} kWh | LGBM Error: {row['err_LightGBM']:6.2f} kWh | EMSformer Error: {row['err_EMSformer']:6.2f} kWh")

print("\n" + "="*80)
print("AUDIT 2: PAPER-BY-PAPER MANUAL VERIFICATION SAMPLES (WHERE OUR MODEL WINS & LOSES)")
print("="*80)

papers = ['LightGBM', 'DLinear', 'TCN_QRNN', 'LTE_Hybrid', 'iTransformer', 'TimeMixer', 'PatchTST', 'EMSformer']

for p in papers:
    print(f"\n>>> COMPARISON: CHAMPION vs {p}")
    diff = base['err_' + p] - base['err_Champion']
    
    # Where Champion wins the most
    best_win = base.loc[diff.idxmax()]
    win_ts = diff.idxmax()
    print(f"  [WHERE OUR MODEL WINS BIGGEST]:")
    print(f"    Timestamp:    {win_ts} (Day: {win_ts.strftime('%A')}, Hour: {win_ts.strftime('%H:%M')})")
    print(f"    Actual Load:  {best_win['actual_kWh']:7.2f} kWh")
    print(f"    Champion:     {best_win['pred_Champion']:7.2f} kWh (Error: {best_win['err_Champion']:6.2f} kWh)")
    print(f"    {p:<12}: {best_win['pred_' + p]:7.2f} kWh (Error: {best_win['err_' + p]:6.2f} kWh)")
    print(f"    Difference:   Champion outperformed {p} by {best_win['err_' + p] - best_win['err_Champion']:6.2f} kWh")

    # Where Paper wins (where diff is most negative)
    best_loss = base.loc[diff.idxmin()]
    loss_ts = diff.idxmin()
    print(f"  [WHERE {p} WAS BETTER (OUR MODEL LOSS / GAP)]:")
    print(f"    Timestamp:    {loss_ts} (Day: {loss_ts.strftime('%A')}, Hour: {loss_ts.strftime('%H:%M')})")
    print(f"    Actual Load:  {best_loss['actual_kWh']:7.2f} kWh")
    print(f"    Champion:     {best_loss['pred_Champion']:7.2f} kWh (Error: {best_loss['err_Champion']:6.2f} kWh)")
    print(f"    {p:<12}: {best_loss['pred_' + p]:7.2f} kWh (Error: {best_loss['err_' + p]:6.2f} kWh)")
    print(f"    Difference:   {p} beat Champion by {best_loss['err_Champion'] - best_loss['err_' + p]:6.2f} kWh")
