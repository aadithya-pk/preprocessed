import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    'n_households',
    'temperature',
    'apparentTemperature',
    'humidity',
    'pressure',
    'hour',
    'dayofweek',
    'is_weekend',
    'is_holiday',
    'month',
    'sin_hour',
    'cos_hour',
    'sin_month',
    'cos_month',
    'lag_1',
    'lag_2',
    'lag_48',
    'lag_336',
    'rolling_mean_48'
]

TARGET_COL = 'grid_load_kWh'

def load_tabular_data(data_dir=r"D:\dataset\smart london\processed_grid_data\2_model_ready_splits"):
    """Loads tabular train, val, and test splits with fitted scaler on train."""
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col='tstp', parse_dates=['tstp'])
    val_df = pd.read_csv(os.path.join(data_dir, "val.csv"), index_col='tstp', parse_dates=['tstp'])
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col='tstp', parse_dates=['tstp'])

    X_train = train_df[FEATURE_COLS].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)

    X_val = val_df[FEATURE_COLS].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.float32)

    X_test = test_df[FEATURE_COLS].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    # Standardize features (fitted strictly on train)
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)

    # Standardize target for neural nets
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

    return {
        'train_df': train_df, 'val_df': val_df, 'test_df': test_df,
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'X_train_scaled': X_train_scaled, 'y_train_scaled': y_train_scaled,
        'X_val_scaled': X_val_scaled, 'y_val_scaled': y_val_scaled,
        'X_test_scaled': X_test_scaled, 'y_test_scaled': y_test_scaled,
        'scaler_X': scaler_X, 'scaler_y': scaler_y,
        'feature_names': FEATURE_COLS
    }

class TimeSeriesSequenceDataset(Dataset):
    """Generates sliding sequence windows of length L to predict next step or horizon."""
    def __init__(self, X, y, lookback=48):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.lookback = lookback

    def __len__(self):
        return len(self.X) - self.lookback

    def __getitem__(self, idx):
        # Sequence of shape (lookback, num_features)
        x_seq = self.X[idx : idx + self.lookback]
        # Target at the end of the window
        y_target = self.y[idx + self.lookback]
        return x_seq, y_target

def get_sequence_loaders(lookback=48, batch_size=64):
    data = load_tabular_data()
    train_dataset = TimeSeriesSequenceDataset(data['X_train_scaled'], data['y_train_scaled'], lookback=lookback)
    val_dataset = TimeSeriesSequenceDataset(data['X_val_scaled'], data['y_val_scaled'], lookback=lookback)
    test_dataset = TimeSeriesSequenceDataset(data['X_test_scaled'], data['y_test_scaled'], lookback=lookback)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler_y': data['scaler_y'],
        'y_test_unscaled': data['y_test'][lookback:],
        'test_index': data['test_df'].index[lookback:],
        'num_features': len(FEATURE_COLS),
        'lookback': lookback
    }
