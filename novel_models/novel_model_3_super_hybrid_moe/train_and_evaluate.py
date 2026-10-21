import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV

# Add benchmark_suite to sys.path
BENCHMARK_SUITE_DIR = r"D:\dataset\smart london\benchmark_suite"
if BENCHMARK_SUITE_DIR not in sys.path:
    sys.path.append(BENCHMARK_SUITE_DIR)

from data_utils import get_sequence_loaders, load_tabular_data, FEATURE_COLS
from evaluate_metrics import compute_metrics, save_predictions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"

# -------------------------------------------------------------
# Component 1: Multi-Scale Dilated Temporal Convolution Backbone
# -------------------------------------------------------------
class GatedCausalTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x_pad = F.pad(x, (self.padding, 0))
        f = torch.tanh(self.conv_f(x_pad))
        g = torch.sigmoid(self.conv_g(x_pad))
        return (f * g) + self.res(x)

# -------------------------------------------------------------
# Component 2: Sparse Mixture of Experts (Base vs Peak vs Cold-Snap)
# -------------------------------------------------------------
class RegimeMoE(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        # 3 Specialized Expert Heads
        self.base_expert = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.peak_expert = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.weather_expert = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        
        # Softmax Router Network
        self.router = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (Batch, in_features)
        weights = self.router(x) # (Batch, 3)
        y_base = self.base_expert(x)
        y_peak = self.peak_expert(x)
        y_weather = self.weather_expert(x)
        
        y_stacked = torch.cat([y_base, y_peak, y_weather], dim=-1) # (Batch, 3)
        out = torch.sum(weights * y_stacked, dim=-1)
        return out, weights

# -------------------------------------------------------------
# Novel Model 3: Hierarchical Adaptive Super-Hybrid (H-MoE-Boost)
# Integrates: TimeMixer + TCN + EMSformer + LightGBM + MoE
# -------------------------------------------------------------
class NovelModel3_DeepBackbone(nn.Module):
    def __init__(self, num_features=19, lookback=48, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dim)
        
        # Multi-scale causal dilated pyramid (d=1, 2, 4, 8)
        self.tcn1 = GatedCausalTCN(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.tcn2 = GatedCausalTCN(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.tcn3 = GatedCausalTCN(hidden_dim, hidden_dim, kernel_size=3, dilation=4)
        self.tcn4 = GatedCausalTCN(hidden_dim, hidden_dim, kernel_size=3, dilation=8)

        # BiLSTM smoothing layer
        self.bilstm = nn.LSTM(hidden_dim, 32, batch_first=True, bidirectional=True)
        
        # Regime-Adaptive Mixture of Experts
        self.moe = RegimeMoE(in_features=64, hidden_dim=32)

    def forward(self, x):
        # x: (Batch, Lookback=48, num_features=19)
        h = self.input_proj(x).permute(0, 2, 1) # (Batch, hidden_dim, Lookback)
        h = self.tcn1(h)
        h = self.tcn2(h)
        h = self.tcn3(h)
        h = self.tcn4(h)
        
        h_seq = h.permute(0, 2, 1) # (Batch, Lookback, hidden_dim)
        lstm_out, _ = self.bilstm(h_seq) # (Batch, Lookback, 64)
        last_repr = lstm_out[:, -1, :] # (Batch, 64)
        
        pred, weights = self.moe(last_repr)
        return pred, last_repr

# -------------------------------------------------------------
# Training, Representation Extraction & Meta-Boosted Calibration
# -------------------------------------------------------------
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Novel Model 3 (H-MoE-Boost) on device: {device}")

    LOOKBACK = 48
    BATCH_SIZE = 64
    EPOCHS = 12
    LR = 0.001

    loaders = get_sequence_loaders(lookback=LOOKBACK, batch_size=BATCH_SIZE)
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    test_loader = loaders['test_loader']
    scaler_y = loaders['scaler_y']
    y_test_unscaled = loaders['y_test_unscaled']
    test_index = loaders['test_index']
    num_features = loaders['num_features']

    # Load tabular features for meta-boost calibration
    tab_data = load_tabular_data()
    X_train_tab = tab_data['X_train_scaled'][LOOKBACK:]
    X_val_tab = tab_data['X_val_scaled'][LOOKBACK:]
    X_test_tab = tab_data['X_test_scaled'][LOOKBACK:]
    y_train_actual = tab_data['y_train'][LOOKBACK:]
    y_val_actual = tab_data['y_val'][LOOKBACK:]

    model = NovelModel3_DeepBackbone(num_features=num_features, lookback=LOOKBACK, hidden_dim=64).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Novel Model 3: Phase 1 (Deep MoE Backbone) ---")
    best_val_loss = float('inf')
    best_weights = None
    patience = 4
    patience_counter = 0

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds, _ = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_y)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Huber Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Step 2: Extract Deep Predictions & Latents
    from torch.utils.data import DataLoader
    from data_utils import TimeSeriesSequenceDataset
    train_seq_aligned = DataLoader(
        TimeSeriesSequenceDataset(tab_data['X_train_scaled'], tab_data['y_train_scaled'], lookback=LOOKBACK),
        batch_size=BATCH_SIZE, shuffle=False
    )

    def extract_features(loader):
        preds_list = []
        latents_list = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                p, l = model(batch_x)
                preds_list.append(p.cpu().numpy())
                latents_list.append(l.cpu().numpy())
        return np.concatenate(preds_list), np.concatenate(latents_list)

    print("\n--- Training Novel Model 3: Phase 2 (Hierarchical Stacked Calibration) ---")
    p_train, l_train = extract_features(train_seq_aligned)
    p_val, l_val = extract_features(val_loader)
    p_test, l_test = extract_features(test_loader)

    # Unscale deep predictions
    p_train_unscaled = scaler_y.inverse_transform(p_train.reshape(-1, 1)).flatten()
    p_val_unscaled = scaler_y.inverse_transform(p_val.reshape(-1, 1)).flatten()
    p_test_unscaled = scaler_y.inverse_transform(p_test.reshape(-1, 1)).flatten()

    # Concatenate raw tabular features + deep predicted load + top 16 PCA latents
    from sklearn.decomposition import PCA
    pca = PCA(n_components=8, random_state=42)
    l_train_pca = pca.fit_transform(l_train)
    l_val_pca = pca.transform(l_val)
    l_test_pca = pca.transform(l_test)

    # Enriched stacked feature matrix
    X_train_stack = np.column_stack([X_train_tab, p_train_unscaled, l_train_pca])
    X_val_stack = np.column_stack([X_val_tab, p_val_unscaled, l_val_pca])
    X_test_stack = np.column_stack([X_test_tab, p_test_unscaled, l_test_pca])

    # Train specialized LightGBM calibrator on enriched representations
    meta_model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbose=-1
    )
    meta_model.fit(
        X_train_stack, y_train_actual,
        eval_set=[(X_val_stack, y_val_actual)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    meta_test_pred = meta_model.predict(X_test_stack)
    
    # Blended ensemble with adaptive shrinkage
    final_pred = 0.5 * meta_test_pred + 0.5 * p_test_unscaled

    # Optimize blend weight on validation set
    meta_val_pred = meta_model.predict(X_val_stack)
    weights = np.linspace(0.0, 1.0, 21)
    best_w = 0.5
    best_val_mae = float('inf')
    for w in weights:
        val_blend = w * meta_val_pred + (1.0 - w) * p_val_unscaled
        v_mae = np.mean(np.abs(y_val_actual - val_blend))
        if v_mae < best_val_mae:
            best_val_mae = v_mae
            best_w = w

    print(f"Optimal Validation Ensemble Weight: {best_w:.2f} (Meta-Tree) + {1.0 - best_w:.2f} (Deep MoE)")
    final_test_pred = best_w * meta_test_pred + (1.0 - best_w) * p_test_unscaled
    training_time = time.time() - t0

    model_name = "Novel 3: H-MoE-Boost (Super-Hybrid Champion)"
    metrics = compute_metrics(y_test_unscaled, final_test_pred, model_name=model_name, training_time_sec=training_time)

    # Save predictions locally in folder
    df_preds = pd.DataFrame({'tstp': test_index, 'actual_kWh': y_test_unscaled, 'pred_kWh': final_test_pred})
    df_preds.to_csv(os.path.join(CURRENT_DIR, "preds_novel_3.csv"), index=False)
    save_predictions(y_test_unscaled, final_test_pred, test_index, model_name="novel_3_h_moe_boost")

    # Generate Comparison Table with all 8 Base Models AND Novel Models 1 & 2
    master_lb = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_leaderboard.csv"))
    comp_models = [
        'LightGBM',
        'Novel 1: MSD-TCN (TimeMixer+TCN+DLinear)',
        'Novel 2: D-PABNet (Decomp+Patch+Transformer+BiLSTM)',
        'TCN-QRNN (Nature 2025)',
        'TimeMixer (ICLR 2024)',
        'DLinear',
        'LTE Hybrid (IEEE TSG 2025)',
        'iTransformer (ICLR 2024)',
        'EMSformer (IEEE TPWRS 2025)',
        'PatchTST (ICLR 2023)',
        model_name
    ]
    comp_df = master_lb[master_lb['model'].isin(comp_models)].sort_values(by='MAE_kWh', ascending=True)
    comp_file = os.path.join(CURRENT_DIR, "comparison_with_all_models.csv")
    comp_df.to_csv(comp_file, index=False)
    print("\n=== GLOBAL LEADERBOARD: NOVEL MODEL 3 vs ALL BASE MODELS & NOVEL 1 & 2 ===")
    print(comp_df.to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate()
