import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from data_utils import get_sequence_loaders
from evaluate_metrics import compute_metrics, save_predictions

class SeriesDecomp(nn.Module):
    """Moving average block to extract trend and seasonal components (AAAI 2023)."""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        # x shape: (Batch, Length, Channels)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        
        # Permute for 1D convolution/pooling: (B, C, L)
        x_perm = x_padded.permute(0, 2, 1)
        trend = self.avg(x_perm).permute(0, 2, 1)
        seasonal = x - trend
        return seasonal, trend

class DLinear(nn.Module):
    """
    DLinear: Decomposition Linear Model for Time Series Forecasting.
    Published at AAAI 2023 by Zeng et al.
    """
    def __init__(self, lookback=48, num_features=19, kernel_size=25):
        super().__init__()
        self.decompsition = SeriesDecomp(kernel_size)
        
        # Flattening lookback * features to predict the single next target step
        in_dim = lookback * num_features
        self.linear_seasonal = nn.Linear(in_dim, 1)
        self.linear_trend = nn.Linear(in_dim, 1)

    def forward(self, x):
        # x: (Batch, Lookback, Features)
        seasonal_init, trend_init = self.decompsition(x)
        
        s_flat = seasonal_init.reshape(seasonal_init.size(0), -1)
        t_flat = trend_init.reshape(trend_init.size(0), -1)
        
        seasonal_output = self.linear_seasonal(s_flat)
        trend_output = self.linear_trend(t_flat)
        
        x_out = seasonal_output + trend_output
        return x_out.squeeze(-1)

def train_and_evaluate_dlinear():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running DLinear on device: {device}")

    LOOKBACK = 48
    BATCH_SIZE = 64
    EPOCHS = 20
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

    model = DLinear(lookback=LOOKBACK, num_features=num_features, kernel_size=25).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Model 2: DLinear (AAAI 2023) ---")
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
    # Unscale predictions back to real physical kWh units
    preds_unscaled = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

    # Compute metrics & update leaderboard
    compute_metrics(y_test_unscaled, preds_unscaled, model_name="DLinear", training_time_sec=training_time)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="DLinear")

if __name__ == "__main__":
    train_and_evaluate_dlinear()
