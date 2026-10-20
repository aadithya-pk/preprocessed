import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_utils import get_sequence_loaders, load_tabular_data, FEATURE_COLS
from evaluate_metrics import compute_metrics, save_predictions

# -------------------------------------------------------------------------
# Component 1: Causal Dilated Conv Block with Weight Normalization & Gating
# -------------------------------------------------------------------------
class GatedDilatedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv_filter = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv_gate = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.res_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # x: (Batch, Channels, Seq_Len)
        # Pad on left only for strict temporal causality
        x_padded = F.pad(x, (self.padding, 0))
        filter_out = torch.tanh(self.conv_filter(x_padded))
        gate_out = torch.sigmoid(self.conv_gate(x_padded))
        gated = filter_out * gate_out
        gated = self.dropout(gated)
        return gated + self.res_proj(x)

# -------------------------------------------------------------------------
# Component 2: Cross-Variate Exogenous Weather & Calendar Gating Unit
# -------------------------------------------------------------------------
class WeatherCalendarGating(nn.Module):
    """
    Explicitly couples meteorological drivers (temperature, humidity, pressure)
    with cyclical time indices (sin_hour, cos_hour, weekend) via gating.
    Resolves the Channel-Independence breakdown in residential load forecasting.
    """
    def __init__(self, num_features=19, embed_dim=64):
        super().__init__()
        self.feat_encoder = nn.Linear(num_features, embed_dim)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        h = self.feat_encoder(x)
        g = self.gate(h)
        out = self.norm(h * g)
        return out

# -------------------------------------------------------------------------
# Component 3: Novel Multi-Scale Dilated TCN with Peak-Aware Attention
# -------------------------------------------------------------------------
class NovelMSSTLFNet(nn.Module):
    """
    Novel MS-STLFNet:
    Multi-Scale Dilated Causal Convolutional Network with Exogenous Weather Gating
    and Peak-Aware Temporal Self-Attention.
    """
    def __init__(self, num_features=19, hidden_dim=64, dilations=[1, 2, 4, 8, 16], n_heads=4, dropout=0.1):
        super().__init__()
        self.exog_gating = WeatherCalendarGating(num_features=num_features, embed_dim=hidden_dim)
        
        # Multi-scale dilated causal convolutional pyramid
        self.tcn_layers = nn.ModuleList([
            GatedDilatedConv1D(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                dilation=d,
                dropout=dropout
            )
            for d in dilations
        ])
        
        # Peak-Aware Multi-Head Self Attention across the final temporal window
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.norm_attn = nn.LayerNorm(hidden_dim)
        
        # Dual-branch Prediction Head: Base Load (smooth) + Peak Surge (high-frequency)
        self.base_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.peak_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        self.combiner = nn.Linear(2, 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        # 1. Weather & Calendar Exogenous Gating
        h = self.exog_gating(x) # (Batch, Seq_Len, hidden_dim)
        
        # 2. Multi-Scale Dilated Convolutions
        h_conv = h.permute(0, 2, 1) # (Batch, hidden_dim, Seq_Len)
        for layer in self.tcn_layers:
            h_conv = layer(h_conv)
        h_conv = h_conv.permute(0, 2, 1) # (Batch, Seq_Len, hidden_dim)
        
        # 3. Peak-Aware Multi-Head Attention
        attn_out, _ = self.attn(h_conv, h_conv, h_conv)
        h_repr = self.norm_attn(h_conv + attn_out) # (Batch, Seq_Len, hidden_dim)
        
        # Last time step summary
        last_step = h_repr[:, -1, :] # (Batch, hidden_dim)
        
        # 4. Dual-Branch Base + Peak Output
        base_out = self.base_head(last_step)
        peak_out = self.peak_head(last_step)
        out = self.combiner(torch.cat([base_out, peak_out], dim=-1))
        return out.squeeze(-1)

# -------------------------------------------------------------------------
# Training, Hybrid Residual Co-Learning & Unified Evaluation
# -------------------------------------------------------------------------
def train_and_evaluate_novel():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Novel MS-STLFNet on device: {device}")

    LOOKBACK = 48
    BATCH_SIZE = 64
    EPOCHS = 12
    LR = 0.001

    print(f"Loading sequence data (Lookback = {LOOKBACK} half-hours / 24h)...")
    loaders = get_sequence_loaders(lookback=LOOKBACK, batch_size=BATCH_SIZE)
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    test_loader = loaders['test_loader']
    scaler_y = loaders['scaler_y']
    y_test_unscaled = loaders['y_test_unscaled']
    test_index = loaders['test_index']
    num_features = loaders['num_features']

    # Also load tabular data for residual tree co-learning
    tab_data = load_tabular_data()
    X_train_tab = tab_data['X_train_scaled'][LOOKBACK:]
    X_val_tab = tab_data['X_val_scaled'][LOOKBACK:]
    X_test_tab = tab_data['X_test_scaled'][LOOKBACK:]

    model = NovelMSSTLFNet(
        num_features=num_features,
        hidden_dim=64,
        dilations=[1, 2, 4, 8, 16],
        n_heads=4,
        dropout=0.1
    ).to(device)

    # Smooth L1 / Huber Loss to be robust to peak spikes and outliers
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    print("\n--- Training Novel MS-STLFNet (Deep Backbone) ---")
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_y)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

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

    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()

    # Step 2: Extract Deep Sequence Predictions on Train, Val, and Test
    def get_all_preds(loader):
        preds_list = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                p = model(batch_x).cpu().numpy()
                preds_list.append(p)
        return np.concatenate(preds_list)

    # Shuffled loader for train was used, but we need aligned sequential predictions for residual boost
    # Create non-shuffled train sequence loader
    from torch.utils.data import DataLoader
    from data_utils import TimeSeriesSequenceDataset
    train_seq_aligned = DataLoader(
        TimeSeriesSequenceDataset(tab_data['X_train_scaled'], tab_data['y_train_scaled'], lookback=LOOKBACK),
        batch_size=BATCH_SIZE, shuffle=False
    )
    
    print("\nComputing deep representations for residual boosting stage...")
    deep_train_preds = get_all_preds(train_seq_aligned)
    deep_val_preds = get_all_preds(val_loader)
    deep_test_preds = get_all_preds(test_loader)

    # Unscale predictions
    y_train_actual = tab_data['y_train'][LOOKBACK:]
    y_val_actual = tab_data['y_val'][LOOKBACK:]
    deep_train_unscaled = scaler_y.inverse_transform(deep_train_preds.reshape(-1, 1)).flatten()
    deep_val_unscaled = scaler_y.inverse_transform(deep_val_preds.reshape(-1, 1)).flatten()
    deep_test_unscaled = scaler_y.inverse_transform(deep_test_preds.reshape(-1, 1)).flatten()

    # Compute deep backbone standalone metrics
    print("\n--- Standalone Deep MS-STLFNet Backbone Metrics ---")
    deep_mae = mean_absolute_error(y_test_unscaled, deep_test_unscaled)
    deep_rmse = np.sqrt(mean_squared_error(y_test_unscaled, deep_test_unscaled))
    print(f"  Deep Backbone MAE : {deep_mae:.4f} kWh | RMSE: {deep_rmse:.4f} kWh")

    # Step 3: Gradient-Boosted Peak Residual Corrector
    # Computes residual error: residual = actual - deep_pred
    train_residual = y_train_actual - deep_train_unscaled
    val_residual = y_val_actual - deep_val_unscaled

    print("\n--- Training Light Residual Tree Corrector ---")
    res_model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    res_model.fit(
        X_train_tab, train_residual,
        eval_set=[(X_val_tab, val_residual)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )

    # Predict residual correction on test set
    pred_test_residual = res_model.predict(X_test_tab)
    
    # Final Ensembled Forecast
    final_test_pred = deep_test_unscaled + pred_test_residual
    total_training_time = time.time() - t0

    # Unified Literature Metric Evaluation
    metrics = compute_metrics(
        y_test_unscaled,
        final_test_pred,
        model_name="Novel MS-STLFNet (Proposed Hybrid)",
        training_time_sec=total_training_time
    )
    save_predictions(y_test_unscaled, final_test_pred, test_index, model_name="Novel_MS_STLFNet")

    # Also log standalone deep backbone
    compute_metrics(
        y_test_unscaled,
        deep_test_unscaled,
        model_name="MS-STLFNet Backbone (Ablation: No Residual Tree)",
        training_time_sec=total_training_time
    )

if __name__ == "__main__":
    train_and_evaluate_novel()
