import time
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

from data_utils import get_sequence_loaders
from evaluate_metrics import compute_metrics, save_predictions

class PatchTST(nn.Module):
    """
    PatchTST: A Time Series is Worth 64 Words: Long-Term Forecasting with Transformers.
    Published at ICLR 2023 by Nie et al. (IBM Research / Princeton).
    Features: Patching + Channel Independence + Self-Attention across patches.
    """
    def __init__(self, num_features=19, lookback=48, patch_len=16, stride=8, d_model=128, n_heads=4, num_layers=3, ffn_dim=256, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_features = num_features
        self.lookback = lookback

        # Number of patches: N = (lookback - patch_len) // stride + 1
        self.num_patches = (lookback - patch_len) // stride + 1

        # 1. Linear Projection of each patch of length P to d_model
        self.patch_embedding = nn.Linear(patch_len, d_model)

        # 2. Learnable Positional Encoding across patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model) * 0.02)

        # 3. Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Flatten + Linear Head (Channel-Independent projection)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.num_patches * d_model * num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (Batch, Lookback, Features)
        # Instance normalization (RevIN) per sequence
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-5
        x_norm = (x - mean) / std

        # Unfold into patches: (Batch, Features, Lookback) -> patches of size patch_len
        x_perm = x_norm.permute(0, 2, 1) # (Batch, Features, Lookback)
        patches = x_perm.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # patches shape: (Batch, Features, Num_Patches, Patch_Len)

        batch_size, n_feats, n_patches, p_len = patches.shape
        # Flatten batch and features to process channels in parallel:
        patches_flat = patches.reshape(batch_size * n_feats, n_patches, p_len)
        
        # Patch linear embedding: (Batch*Features, Num_Patches, d_model)
        tokens = self.patch_embedding(patches_flat) + self.pos_embedding
        
        # Self-attention across patches
        enc_out = self.transformer_encoder(tokens) # (Batch*Features, Num_Patches, d_model)

        # Reshape back to (Batch, Features, Num_Patches, d_model)
        enc_out = enc_out.reshape(batch_size, n_feats, n_patches, -1)
        
        # Final head
        pred = self.head(enc_out)
        return pred.squeeze(-1)

def train_and_evaluate_patchtst():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running PatchTST on device: {device}")

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

    model = PatchTST(
        num_features=num_features,
        lookback=LOOKBACK,
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=4,
        num_layers=2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=LR)

    print("\n--- Training Model 7: PatchTST (ICLR 2023) ---")
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
    compute_metrics(y_test_unscaled, preds_unscaled, model_name="PatchTST (ICLR 2023)", training_time_sec=training_time)
    save_predictions(y_test_unscaled, preds_unscaled, test_index, model_name="PatchTST")

if __name__ == "__main__":
    train_and_evaluate_patchtst()
