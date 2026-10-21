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
# Component 1: Series Decomposition (DLinear / TimeMixer)
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
# Component 2: Dilated Causal Convolution (TCN-QRNN)
# -------------------------------------------------------------
class CausalTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.res = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.GELU()

    def forward(self, x):
        # x: (Batch, Channels, Seq_Len)
        # Pad on left only for temporal causality
        x_pad1 = F.pad(x, (self.padding, 0))
        out = self.relu(self.conv1(x_pad1))
        out = self.dropout(out)
        x_pad2 = F.pad(out, (self.padding, 0))
        out = self.conv2(x_pad2)
        out = self.dropout(out)
        return self.relu(out + self.res(x))

# -------------------------------------------------------------
# Novel Model 1: MultiScale-Decomp-TCN (MSD-TCN)
# Combines TimeMixer (ICLR 2024) + TCN-QRNN (Nature 2025) + DLinear (AAAI 2023)
# -------------------------------------------------------------
class NovelModel1_MSD_TCN(nn.Module):
    def __init__(self, num_features=19, seq_len=48, hidden_dim=64, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.decomp = SeriesDecomp(kernel_size=15)

        # Multi-scale downsampling pools (Scale 0: 48, Scale 1: 24, Scale 2: 12)
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2**i, stride=2**i) if i > 0 else nn.Identity()
            for i in range(num_scales)
        ])

        # Causal TCN blocks for each scale's seasonal component
        self.tcn_scales = nn.ModuleList([
            nn.Sequential(
                CausalTCNBlock(num_features, hidden_dim, kernel_size=3, dilation=1),
                CausalTCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
            )
            for _ in range(num_scales)
        ])

        # Linear projections for trend component across scales (DLinear)
        self.trend_layers = nn.ModuleList([
            nn.Linear(seq_len // (2**i), 1)
            for i in range(num_scales)
        ])

        # Seasonal projection per scale
        self.seasonal_layers = nn.ModuleList([
            nn.Linear(seq_len // (2**i), 1)
            for i in range(num_scales)
        ])

        # Fusion layer across scales and components
        self.fusion = nn.Sequential(
            nn.Linear(num_scales * (hidden_dim + num_features), 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len=48, num_features=19)
        scale_features = []

        for i in range(self.num_scales):
            # Downsample to scale
            if i == 0:
                x_scale = x
            else:
                x_scale = self.pools[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            
            # Decompose into Seasonal and Trend
            s_scale, t_scale = self.decomp(x_scale)
            
            # 1. Seasonal processing with Causal TCN
            # s_scale: (Batch, Length_i, Features) -> (Batch, Features, Length_i)
            s_tcn = self.tcn_scales[i](s_scale.permute(0, 2, 1)) # (Batch, hidden_dim, Length_i)
            s_pooled = self.seasonal_layers[i](s_tcn).squeeze(-1) # (Batch, hidden_dim)

            # 2. Trend processing with Linear Projection (DLinear style)
            # t_scale: (Batch, Length_i, Features) -> (Batch, Features, Length_i)
            t_proj = self.trend_layers[i](t_scale.permute(0, 2, 1)).squeeze(-1) # (Batch, num_features)

            scale_features.append(s_pooled)
            scale_features.append(t_proj)

        # Concatenate all scale features
        all_features = torch.cat(scale_features, dim=-1) # (Batch, num_scales * (hidden_dim + num_features))
        out = self.fusion(all_features)
        return out.squeeze(-1)

# -------------------------------------------------------------
# Training & Evaluation Script
# -------------------------------------------------------------
def train_and_evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Novel Model 1 (MSD-TCN) on device: {device}")

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

    model = NovelModel1_MSD_TCN(num_features=num_features, seq_len=LOOKBACK, hidden_dim=64, num_scales=3).to(device)
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Novel Model 1: MultiScale-Decomp-TCN (MSD-TCN) ---")
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

    model_name = "Novel 1: MSD-TCN (TimeMixer+TCN+DLinear)"
    metrics = compute_metrics(y_test_unscaled, preds_unscaled, model_name=model_name, training_time_sec=training_time)
    
    # Save predictions locally in folder
    df_preds = pd.DataFrame({'tstp': test_index, 'actual_kWh': y_test_unscaled, 'pred_kWh': preds_unscaled})
    df_preds.to_csv(os.path.join(CURRENT_DIR, "preds_novel_1.csv"), index=False)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="novel_1_msd_tcn")

    # Generate Comparison Table with all 8 Base Models
    master_lb = pd.read_csv(os.path.join(RESULTS_DIR, "benchmark_leaderboard.csv"))
    # Filter to base models + Novel 1
    base_model_names = [
        'LightGBM',
        'TCN-QRNN (Nature 2025)',
        'TimeMixer (ICLR 2024)',
        'DLinear',
        'LTE Hybrid (IEEE TSG 2025)',
        'iTransformer (ICLR 2024)',
        'EMSformer (IEEE TPWRS 2025)',
        'PatchTST (ICLR 2023)',
        model_name
    ]
    comp_df = master_lb[master_lb['model'].isin(base_model_names)].sort_values(by='MAE_kWh', ascending=True)
    comp_file = os.path.join(CURRENT_DIR, "comparison_with_baselines.csv")
    comp_df.to_csv(comp_file, index=False)
    print("\n=== COMPARISON: NOVEL MODEL 1 vs ALL 8 BASE MODELS ===")
    print(comp_df.to_string(index=False))

if __name__ == "__main__":
    train_and_evaluate()
