import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from data_utils import get_sequence_loaders
from evaluate_metrics import compute_metrics, save_predictions

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as defined in Eq. (2) of Pentsos et al. (IEEE TSG 2025)."""
    def __init__(self, d_model=256, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (Batch, Seq_Len, d_model)
        return x + self.pe[:, :x.size(1)]

class LTEHybrid(nn.Module):
    """
    LTE Hybrid Model: LSTM + Transformer Encoder for Power Load Forecasting.
    Published in IEEE Transactions on Smart Grid, Vol. 16, No. 3, May 2025 by Pentsos et al.
    Parameters: K=2 LSTM layers, L=32 cells, T=256 embedding, P=6 encoder layers, R=4 attention heads.
    """
    def __init__(self, num_features=19, lstm_hidden=32, num_lstm_layers=2, embed_dim=256, num_heads=4, num_encoder_layers=6, ffn_dim=1024):
        super().__init__()
        # 1. Front-end: K=2 LSTM layers with L=32 cells
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True
        )
        # Fully connected layer mapping LSTM output to T=256 embeddings
        self.fc_embed = nn.Linear(lstm_hidden, embed_dim)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model=embed_dim)

        # 3. Back-end: P=6 stacked Transformer Encoder layers with R=4 heads
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            dropout=0.1
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 4. Final Linear Regression Head
        self.regressor = nn.Linear(embed_dim, 1)

    def forward(self, x):
        # x: (Batch, Lookback, Features)
        lstm_out, _ = self.lstm(x)                  # (Batch, Lookback, 32)
        embeddings = self.fc_embed(lstm_out)        # (Batch, Lookback, 256)
        embeddings = self.pos_encoder(embeddings)   # Add positional encodings
        trans_out = self.transformer_encoder(embeddings) # Multi-head self-attention
        
        # Pool last time step representation
        last_step = trans_out[:, -1, :]
        pred = self.regressor(last_step)
        return pred.squeeze(-1)

def train_and_evaluate_lte():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running LTE Hybrid on device: {device}")

    LOOKBACK = 48
    BATCH_SIZE = 64
    EPOCHS = 15
    LR = 0.0005

    print(f"Loading sequence data (Lookback = {LOOKBACK} half-hours / 24h)...")
    loaders = get_sequence_loaders(lookback=LOOKBACK, batch_size=BATCH_SIZE)
    train_loader = loaders['train_loader']
    val_loader = loaders['val_loader']
    test_loader = loaders['test_loader']
    scaler_y = loaders['scaler_y']
    y_test_unscaled = loaders['y_test_unscaled']
    test_index = loaders['test_index']
    num_features = loaders['num_features']

    # Pentsos et al. parameters
    model = LTEHybrid(
        num_features=num_features,
        lstm_hidden=32,
        num_lstm_layers=2,
        embed_dim=256,
        num_heads=4,
        num_encoder_layers=4, # 4 layers for balanced training speed on CPU
        ffn_dim=1024
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Model 4: LTE Hybrid (IEEE Trans. Smart Grid, May 2025) ---")
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
    compute_metrics(y_test_unscaled, preds_unscaled, model_name="LTE Hybrid (IEEE TSG 2025)", training_time_sec=training_time)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="LTE_Hybrid")

if __name__ == "__main__":
    train_and_evaluate_lte()
