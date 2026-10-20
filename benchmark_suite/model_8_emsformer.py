import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from data_utils import get_sequence_loaders
from evaluate_metrics import compute_metrics, save_predictions

class PatchEmbedding(nn.Module):
    """
    Patch Embedding Operation as defined in Section II-C and Fig. 4 of Fang et al. (IEEE TPWRS 2025).
    Divides sequence into patches of length N1 with stride S1, projects through linear layer.
    """
    def __init__(self, patch_len=4, stride=1, in_channels=19, embed_dim=128):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        # Projects flattened patch (patch_len * in_channels) to embed_dim
        self.proj = nn.Linear(patch_len * in_channels, embed_dim)

    def forward(self, x):
        # x: (Batch, Lookback, In_Channels)
        # Unfold along time axis
        batch_size, lookback, in_channels = x.shape
        # Unfold requires (Batch, In_Channels, Lookback)
        x_perm = x.permute(0, 2, 1)
        # unfold: (Batch, In_Channels, Num_Patches, Patch_Len)
        patches = x_perm.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        num_patches = patches.size(2)
        
        # Permute to (Batch, Num_Patches, In_Channels, Patch_Len) and flatten features
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, num_patches, in_channels * self.patch_len)
        
        # Linear projection to embed_dim
        embedded_patches = self.proj(patches)
        return embedded_patches # (Batch, Num_Patches, embed_dim)

class EMSformer(nn.Module):
    """
    EMSformer: Transformer-BiLSTM-Patch Architecture for Power Forecasting.
    Published in IEEE Transactions on Power Systems, Vol. 40, No. 6, Nov 2025 by Fang et al.
    Combines Patch Embedding + Unmasked Transformer Attention + BiLSTM backend error correction.
    """
    def __init__(self, in_channels=19, lookback=48, patch_len=4, stride=1, d_model=128, n_heads=4, num_decoder_layers=2, bilstm_hidden=64):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_len=patch_len, stride=stride, in_channels=in_channels, embed_dim=d_model)

        # Unmasked Self-Attention Transformer Decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)

        # Backend BiLSTM for temporal feature extraction and error correction (Section II-D)
        self.bilstm = nn.LSTM(
            input_size=d_model,
            hidden_size=bilstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Output regression head
        self.regressor = nn.Sequential(
            nn.Linear(bilstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (Batch, Lookback, In_Channels)
        # 1. Patch Embedding
        patches = self.patch_embed(x) # (Batch, Num_Patches, d_model)
        
        # 2. Unmasked Decoder Attention (captures medium and long-term global trends)
        dec_out = self.decoder(patches) # (Batch, Num_Patches, d_model)

        # 3. BiLSTM backend layer (captures dynamic temporal features and corrects decoder errors)
        lstm_out, _ = self.bilstm(dec_out) # (Batch, Num_Patches, bilstm_hidden * 2)

        # 4. Final forecast from last step
        last_step = lstm_out[:, -1, :]
        pred = self.regressor(last_step)
        return pred.squeeze(-1)

def train_and_evaluate_emsformer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running EMSformer on device: {device}")

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

    # Fang et al. hyperparameters
    model = EMSformer(
        in_channels=num_features,
        lookback=LOOKBACK,
        patch_len=4,
        stride=1,
        d_model=128,
        n_heads=4,
        num_decoder_layers=2,
        bilstm_hidden=64
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Model 8: EMSformer (IEEE Trans. Power Systems, Nov 2025) ---")
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
    compute_metrics(y_test_unscaled, preds_unscaled, model_name="EMSformer (IEEE TPWRS 2025)", training_time_sec=training_time)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="EMSformer")

if __name__ == "__main__":
    train_and_evaluate_emsformer()
