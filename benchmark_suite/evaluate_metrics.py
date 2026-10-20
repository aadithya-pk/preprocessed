import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
LEADERBOARD_FILE = os.path.join(RESULTS_DIR, "benchmark_leaderboard.csv")

def compute_metrics(y_true, y_pred, model_name, training_time_sec=0.0):
    """
    Computes unified metrics across the union of literature:
    MAE, RMSE, WAPE (%), MAPE (%), R2, and Training Time.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    
    # WAPE (%): Sum(|y - y_hat|) / Sum(y) * 100
    sum_actual = float(np.sum(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / (sum_actual + 1e-9) * 100.0)

    # MAPE (%): filtered against zero denominators
    epsilon = 1e-5
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100.0)

    # R2 Score
    r2 = float(r2_score(y_true, y_pred))

    metrics = {
        'model': model_name,
        'MAE_kWh': round(mae, 4),
        'RMSE_kWh': round(rmse, 4),
        'WAPE_pct': round(wape, 2),
        'MAPE_pct': round(mape, 2),
        'R2_score': round(r2, 4),
        'train_time_sec': round(training_time_sec, 2)
    }

    # Save individual JSON summary
    json_path = os.path.join(RESULTS_DIR, f"{model_name.lower().replace(' ', '_').replace('/', '_')}_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    # Update master CSV leaderboard
    update_leaderboard(metrics)

    print(f"\n=======================================================")
    print(f"RESULTS FOR: {model_name}")
    print(f"=======================================================")
    print(f"  MAE  : {metrics['MAE_kWh']} kWh")
    print(f"  RMSE : {metrics['RMSE_kWh']} kWh")
    print(f"  WAPE : {metrics['WAPE_pct']}%")
    print(f"  R^2  : {metrics['R2_score']}")
    print(f"  Time : {metrics['train_time_sec']}s")
    print(f"=======================================================\n")

    return metrics

def update_leaderboard(new_metrics):
    df_new = pd.DataFrame([new_metrics])
    if os.path.exists(LEADERBOARD_FILE):
        df_old = pd.read_csv(LEADERBOARD_FILE)
        # Drop existing entry for this model if re-run
        df_old = df_old[df_old['model'] != new_metrics['model']]
        df_updated = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_updated = df_new

    # Sort leaderboard by lowest MAE
    df_updated = df_updated.sort_values(by='MAE_kWh', ascending=True)
    df_updated.to_csv(LEADERBOARD_FILE, index=False)
    print(f"Master leaderboard updated at: {LEADERBOARD_FILE}")

def save_predictions(y_true, y_pred, index, model_name):
    """Saves timestamps, ground truth, and predicted load to a CSV for plotting."""
    df_preds = pd.DataFrame({
        'tstp': index,
        'actual_kWh': y_true,
        'pred_kWh': y_pred
    })
    filename = os.path.join(RESULTS_DIR, f"preds_{model_name.lower().replace(' ', '_').replace('/', '_')}.csv")
    df_preds.to_csv(filename, index=False)
    print(f"Predictions saved to: {filename}")
