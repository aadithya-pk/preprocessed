import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import pandas as pd

# Add benchmark_suite to sys.path
BENCHMARK_SUITE_DIR = r"D:\dataset\smart london\benchmark_suite"
if BENCHMARK_SUITE_DIR not in sys.path:
    sys.path.append(BENCHMARK_SUITE_DIR)

from data_utils import get_sequence_loaders, FEATURE_COLS
from evaluate_metrics import compute_metrics, save_predictions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = r"D:\dataset\smart london\benchmark_results"

# -------------------------------------------------------------
# Component 1: Moving Average Decomposition (TimeMixer / DLinear)
# -------------------------------------------------------------
class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size=15):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x: (Batch, Seq_Len, Channels)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend

# -------------------------------------------------------------
# Component 2: Overlapping Patch Embedding (EMSformer / PatchTST)
# -------------------------------------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len=4, stride=2, in_channels=19, embed_dim=64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len * in_channels, embed_dim)

    def forward(self, x):
        # x: (Batch, Lookback=48, In_Channels=19)
        batch_size, lookback, in_channels = x.shape
        x_perm = x.permute(0, 2, 1) # (Batch, In_Channels, Lookback)
        patches = x_perm.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        num_patches = patches.size(2)
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, num_patches, in_channels * self.patch_len)
        return self.proj(patches) # (Batch, Num_Patches, embed_dim)

# -------------------------------------------------------------
# Novel Model 2: Decomposed Patch Attention with BiLSTM Error Correction
# Combines EMSformer (IEEE TPWRS 2025) + TimeMixer (ICLR 2024) + LTE Hybrid (IEEE TSG 2025)
# -------------------------------------------------------------
class NovelModel2_DPABNet(nn.Module):
    def __init__(self, in_channels=19, lookback=48, patch_len=4, stride=2, d_model=64, n_heads=4, bilstm_hidden=64):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size=15)
        
        # Trend linear highway
        self.trend_head = nn.Linear(lookback * in_channels, 32)

        # Seasonal Patch Embedding
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, in_channels=in_channels, embed_dim=d_model)

        # Unmasked Transformer Encoder Layer (EMSformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # BiLSTM backend for dynamic sequential error correction (LTE + EMSformer)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=bilstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Final Regressor fusing Trend + Seasonal BiLSTM
        self.fusion = nn.Sequential(
            nn.Linear(32 + bilstm_hidden * 2, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, Lookback=48, Features=19)
        # 1. Decompose Trend and Seasonal
        seasonal, trend = self.decomp(x)

        # 2. Trend Highway
        trend_flat = trend.reshape(trend.size(0), -1)
        trend_feat = self.trend_head(trend_flat) # (Batch, 32)

        # 3. Seasonal Overlapping Patch Attention
        patches = self.patch_embed(seasonal) # (Batch, Num_Patches, d_model)
        attn_out = self.transformer(patches) # (Batch, Num_Patches, d_model)

        # 4. BiLSTM dynamic error correction
        lstm_out, _ = self.bilstm(attn_out) # (Batch, Num_Patches, bilstm_hidden * 2)
        seasonal_feat = lstm_out[:, -1, :] # (Batch, bilstm_hidden * 2)

        # 5. Fusion
        combined = torch.cat([trend_feat, seasonal_feat], dim=-1)
        out = self.fusion(combined)
        return out.squeeze(-1)

# -------------------------------------------------------------
# Training & Evaluation Script
# -------------------------------------------------------------
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Novel Model 2 (D-PABNet) on device: {device}")

    LOOKBACK = 48
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 0.001

    loaders = get_sequence_loaders(lookback=LOOKBACK, batch_size=BATCH_SIZE)
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    test_loader = loaders['test_loader']
    scaler_y = loaders['scaler_y']
    y_test_unscaled = loaders['y_test_unscaled']
    test_index = loaders['test_index']
    num_features = loaders['num_features']

    model = NovelModel2_DPABNet(
        in_channels=num_features,
        lookback=LOOKBACK,
        patch_len=4,
        stride=2,
        d_model=64,
        n_heads=4,
        bilstm_hidden=64
    ).to(device)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Novel Model 2: Decomposed Patch-Attention BiLSTM (D-PABNet) ---")
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
            preds = model(batch_x)
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
                preds = model(batch_x)
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

    training_time = time.time() - t0

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Evaluation on test set
    preds_scaled_list = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            p = model(batch_x).cpu().numpy()
            preds_scaled_list.append(p)

    preds_scaled = np.concatenate(preds_scaled_list)
    preds_unscaled = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    model_name = "Novel 2: D-PABNet (Decomp+Patch+Transformer+BiLSTM)"
    metrics = compute_metrics(y_test_unscaled, preds_unscaled, model_name=model_name, training_time_sec=training_time)
    
    # Save predictions locally in folder
    df_preds = pd.DataFrame({'tstp': test_index, 'actual_kWh': y_test_unscaled, 'pred_kWh': preds_unscaled})
    df_preds.to_csv(os.path.join(CURRENT_DIR, "preds_novel_2.csv"), index=False)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="novel_2_dpabnet")

    # Generate Comparison Table with all 8 Base Models AND Novel Model 1
    master_lb = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_leaderboard.csv"))
    comp_models = [
        'LightGBM',
        'Novel 1: MSD-TCN (TimeMixer+TCN+DLinear)',
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
    comp_file = os.path.join(CURRENT_DIR, "comparison_with_baselines_and_novel1.csv")
    comp_df.to_csv(comp_file, index=False)
    print("\n=== COMPARISON: NOVEL MODEL 2 vs ALL BASE MODELS & NOVEL MODEL 1 ===")
    print(comp_df.to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate()
