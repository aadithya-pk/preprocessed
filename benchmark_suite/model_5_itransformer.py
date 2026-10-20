import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from data_utils import get_sequence_loaders
from evaluate_metrics import compute_metrics, save_predictions

class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformers Are Effective for Time Series Forecasting.
    Published at ICLR 2024 (Spotlight) by Liu et al. (Tsinghua University).
    Inverts the dimensions: variables are tokens, attention captures cross-variable correlations.
    """
    def __init__(self, num_features=19, lookback=48, d_model=128, n_heads=4, num_layers=2, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.lookback = lookback
        
        # 1. Project each series of length L into a d_model token
        self.enc_embedding = nn.Linear(lookback, d_model)

        # 2. Inverted Transformer Encoder (Self-Attention across N variables)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Output Projection: maps token dimension d_model to 1 prediction step
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x):
        # x input: (Batch, Lookback, Num_Features)
        # Invert dimensions: (Batch, Num_Features, Lookback)
        x_inv = x.permute(0, 2, 1)

        # Embed each variate's whole series as a token: (Batch, Num_Features, d_model)
        tokens = self.enc_embedding(x_inv)

        # Self-attention over variates (captures multivariate correlations)
        attn_out = self.transformer_encoder(tokens)

        # Project tokens: (Batch, Num_Features, 1)
        pred_all_vars = self.projection(attn_out).squeeze(-1) # (Batch, Num_Features)

        # We predict target variable (grid_load_kWh, which is target index or pooled representation)
        # In our tabular setup, let's take a learned linear combination over the variates
        return pred_all_vars[:, 0] # First feature is grid load / target

class iTransformerWithHead(nn.Module):
    def __init__(self, num_features=19, lookback=48, d_model=128, n_heads=4, num_layers=2):
        super().__init__()
        # Project each variate to d_model
        self.enc_embedding = nn.Linear(lookback, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head pools variate tokens to predict target
        self.out_head = nn.Sequential(
            nn.Linear(num_features * d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (Batch, Lookback, Num_Features) -> Invert: (Batch, Num_Features, Lookback)
        x_inv = x.permute(0, 2, 1)
        tokens = self.enc_embedding(x_inv) # (Batch, Num_Features, d_model)
        tokens = self.transformer_encoder(tokens) # Attention across variates
        flat = tokens.reshape(tokens.size(0), -1)
        pred = self.out_head(flat)
        return pred.squeeze(-1)

def train_and_evaluate_itransformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running iTransformer on device: {device}")

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

    model = iTransformerWithHead(num_features=num_features, lookback=LOOKBACK, d_model=128, n_heads=4, num_layers=2).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Model 5: iTransformer (ICLR 2024 Spotlight) ---")
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
    compute_metrics(y_test_unscaled, preds_unscaled, model_name="iTransformer (ICLR 2024)", training_time_sec=training_time)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="iTransformer")

if __name__ == "__main__":
    train_and_evaluate_itransformer()
