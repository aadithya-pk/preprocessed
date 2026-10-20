import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from data_utils import get_sequence_loaders
from evaluate_metrics import compute_metrics, save_predictions

class ChausalConv1d(nn.Module):
    """Causal 1D convolution that prevents future information leakage."""
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        if self.padding != 0:
            out = out[:, :, :-self.padding]
        return out

class TCNBlock(nn.Module):
    """Dilated Temporal Convolutional Network block with residual connection."""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        return nn.functional.relu(out + residual)

class QRNNLayer(nn.Module):
    """
    Quasi-Recurrent Neural Network layer (Bradbury et al., 2016; Mochurad & Levkovych, Nature 2025).
    Uses 1D convolution for parallel feature calculation, followed by fast f-fo-pooling.
    """
    def __init__(self, in_channels, hidden_size, kernel_size=3):
        super().__init__()
        self.hidden_size = hidden_size
        # Convolutions for candidate (c), forget gate (f), and output gate (o)
        self.conv_c = ChausalConv1d(in_channels, hidden_size, kernel_size)
        self.conv_f = ChausalConv1d(in_channels, hidden_size, kernel_size)
        self.conv_o = ChausalConv1d(in_channels, hidden_size, kernel_size)

    def forward(self, x):
        # x: (Batch, Channels, Length)
        c = torch.tanh(self.conv_c(x))
        f = torch.sigmoid(self.conv_f(x))
        o = torch.sigmoid(self.conv_o(x))

        # Recurrent fo-pooling across time dimension
        batch_size, _, seq_len = c.size()
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_state = torch.zeros(batch_size, self.hidden_size, device=x.device)

        h_list = []
        for t in range(seq_len):
            c_t = c[:, :, t]
            f_t = f[:, :, t]
            o_t = o[:, :, t]
            c_state = f_t * c_state + (1.0 - f_t) * c_t
            h_t = o_t * c_state
            h_list.append(h_t.unsqueeze(-1))

        h_seq = torch.cat(h_list, dim=-1)
        return h_seq

class TCN_QRNN(nn.Module):
    """
    The proposed TCN-QRNN architecture from Nature Scientific Reports (2025).
    Stacks multi-layer dilated TCN + QRNN + SiLU dense layer + output regressor.
    """
    def __init__(self, num_features=19, hidden_size=64):
        super().__init__()
        # 1. TCN Stage (Dilated causal conv layers)
        self.tcn1 = TCNBlock(num_features, 64, kernel_size=3, dilation=1)
        self.tcn2 = TCNBlock(64, 64, kernel_size=3, dilation=2)
        self.tcn3 = TCNBlock(64, 64, kernel_size=3, dilation=4)

        # 2. QRNN Stage
        self.qrnn = QRNNLayer(64, hidden_size=hidden_size, kernel_size=3)

        # 3. Dense with SiLU activation (as specified in Nature paper)
        self.fc_silu = nn.Linear(hidden_size, 64)
        self.silu = nn.SiLU()

        # 4. Final output layer
        self.regressor = nn.Linear(64, 1)

    def forward(self, x):
        # Input x: (Batch, Lookback, Features) -> Permute to (Batch, Features, Lookback)
        x_in = x.permute(0, 2, 1)
        
        # TCN feature extraction
        out = self.tcn1(x_in)
        out = self.tcn2(out)
        out = self.tcn3(out)

        # QRNN sequence modeling
        out = self.qrnn(out)

        # Take last time step
        last_hidden = out[:, :, -1]

        # SiLU dense + output
        dense_out = self.silu(self.fc_silu(last_hidden))
        pred = self.regressor(dense_out)
        return pred.squeeze(-1)

def train_and_evaluate_tcn_qrnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running TCN-QRNN on device: {device}")

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

    model = TCN_QRNN(num_features=num_features, hidden_size=64).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Model 3: TCN-QRNN (Nature Sci. Rep. 2025) ---")
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
    compute_metrics(y_test_unscaled, preds_unscaled, model_name="TCN-QRNN (Nature 2025)", training_time_sec=training_time)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="TCN_QRNN")

if __name__ == "__main__":
    train_and_evaluate_tcn_qrnn()
