#!/usr/bin/env python3
"""
EEG Denoising - Lightweight 1D CNN
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

print("="*60)
print("EEG DENOISING - 1D CNN")
print("="*60)

# Load
print("\n[1] Loading data...")
eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
print(f"    EEG: {eeg.shape}, EOG: {eog.shape}")

# Split
indices = np.arange(2000)
np.random.shuffle(indices)
train_idx = indices[:1700]
test_idx = indices[1700:]

def mix_at_snr(clean, noise, snr_db):
    k = np.sqrt(np.mean(clean**2) / (10**(snr_db/10) * np.mean(noise**2)))
    return clean + k * noise

# Prepare data
print("\n[2] Preparing data...")

def prepare_data(indices, snr_db):
    X, y = [], []
    for idx in indices:
        clean = eeg[idx]
        noise = eog[idx % len(eog)]
        noisy = mix_at_snr(clean, noise, snr_db)
        X.append(noisy)
        y.append(clean)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# Training at -5dB
X_train, y_train = prepare_data(train_idx, -5)
X_test, y_test = prepare_data(test_idx, -5)

# Normalize
x_mean, x_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train = (X_train - x_mean) / x_std
X_test = (X_test - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std

print(f"    Train: {X_train.shape}, Test: {X_test.shape}")

# Simple 1D CNN
class DenoiseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze(1)

# DataLoader
X_t = torch.FloatTensor(X_train).unsqueeze(1)
y_t = torch.FloatTensor(y_train_norm).unsqueeze(1)
loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)

# Train
print("\n[3] Training CNN...")
model = DenoiseCNN()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model.train()
for epoch in range(20):
    total_loss = 0
    for bx, by in loader:
        opt.zero_grad()
        out = model(bx)
        loss = criterion(out, by)
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"    Epoch {epoch+1}/20, Loss: {total_loss/len(loader):.6f}")

# Test
print("\n[4] Testing...")
model.eval()
with torch.no_grad():
    X_te_t = torch.FloatTensor(X_test).unsqueeze(1)
    preds_norm = model(X_te_t).numpy()
    preds = preds_norm * y_std + y_mean

# Evaluate
pearson_scores = []
rrmse_scores = []

for i in range(len(test_idx)):
    c, _ = pearsonr(preds[i], y_test[i])
    mse = np.mean((preds[i] - y_test[i])**2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.sqrt(np.mean(y_test[i]**2))
    pearson_scores.append(c)
    rrmse_scores.append(rrmse)

mean_p = np.mean(pearson_scores)
std_p = np.std(pearson_scores)
mean_r = np.mean(rrmse_scores)
std_r = np.std(rrmse_scores)

print("\n" + "="*60)
print("CNN RESULTS")
print("="*60)
print(f"Test samples: {len(test_idx)}")
print(f"SNR: -5 dB")
print("-"*60)
print(f"TARGET: Pearson >= 0.98, RRMSE <= 0.15")
print("-"*60)
print(f"1D CNN: Pearson={mean_p:.4f} +/- {std_p:.4f}, RRMSE={mean_r:.4f} +/- {std_r:.4f}")
print("="*60)
