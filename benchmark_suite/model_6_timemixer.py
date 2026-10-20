import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from data_utils import get_sequence_loaders
from evaluate_metrics import compute_metrics, save_predictions

class SeriesDecomp(nn.Module):
    """Decomposition block for TimeMixer."""
    def __init__(self, kernel_size=25):
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

class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up multiscale mixing for seasonality."""
    def __init__(self, seq_len, down_sampling_layers=3):
        super().__init__()
        self.mixing_layers = nn.ModuleList([
            nn.Linear(seq_len // (2**i), seq_len // (2**(i+1)))
            for i in range(down_sampling_layers - 1)
        ])

    def forward(self, season_list):
        # season_list: [S_0 (fine), S_1 (medium), S_2 (coarse)]
        out_list = [season_list[0]]
        for i, layer in enumerate(self.mixing_layers):
            s_down = layer(out_list[-1].permute(0, 2, 1)).permute(0, 2, 1)
            out_list.append(s_down + season_list[i+1])
        return out_list

class TimeMixerBlock(nn.Module):
    """
    TimeMixer Block (ICLR 2024 by Wang et al.).
    Decomposable multiscale mixing on both trend and seasonal streams.
    """
    def __init__(self, seq_len=48, num_features=19, down_sampling_layers=3):
        super().__init__()
        self.down_sampling_layers = down_sampling_layers
        self.decomp = SeriesDecomp(kernel_size=15)
        
        # Multiscale average pooling
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=2**i, stride=2**i) if i > 0 else nn.Identity()
            for i in range(down_sampling_layers)
        ])
        
        # Linear projection layers for each scale
        self.linear_layers = nn.ModuleList([
            nn.Linear(seq_len // (2**i), 1)
            for i in range(down_sampling_layers)
        ])
        
        self.feature_combiner = nn.Linear(num_features * down_sampling_layers, 1)

    def forward(self, x):
        # x: (Batch, Seq_Len, Features)
        # 1. Multi-scale downsampling
        scales = []
        for i, pool in enumerate(self.pools):
            if i == 0:
                scales.append(x)
            else:
                x_p = pool(x.permute(0, 2, 1)).permute(0, 2, 1)
                scales.append(x_p)

        # 2. Decompose each scale & project
        scale_outputs = []
        for i, (scale_x, lin) in enumerate(zip(scales, self.linear_layers)):
            seasonal, trend = self.decomp(scale_x)
            # Combine temporal axis: (Batch, Features, Length) -> (Batch, Features, 1)
            s_proj = lin(seasonal.permute(0, 2, 1))
            t_proj = lin(trend.permute(0, 2, 1))
            scale_outputs.append((s_proj + t_proj).squeeze(-1)) # (Batch, Features)

        # 3. Concatenate all multiscale features and predict target
        all_scales = torch.cat(scale_outputs, dim=-1)
        pred = self.feature_combiner(all_scales)
        return pred.squeeze(-1)

def train_and_evaluate_timemixer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TimeMixer on device: {device}")

    LOOKBACK = 48
    BATCH_SIZE = 64
    EPOCHS = 15
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

    model = TimeMixerBlock(seq_len=LOOKBACK, num_features=num_features, down_sampling_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Model 6: TimeMixer (ICLR 2024) ---")
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

        print(f"Epoch {epoch:2d}/{EPOCHS} | Train Loss (scaled MSE): {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
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

    # Compute metrics & update leaderboard
    compute_metrics(y_test_unscaled, preds_unscaled, model_name="TimeMixer (ICLR 2024)", training_time_sec=training_time)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="TimeMixer")

if __name__ == "__main__":
    train_and_evaluate_timemixer()
