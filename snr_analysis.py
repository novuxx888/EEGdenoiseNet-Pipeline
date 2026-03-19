#!/usr/bin/env python3
"""
EEG Denoising - SNR Analysis
See which SNR is achievable
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("="*60)
print("EEG DENOISING - SNR ANALYSIS")
print("="*60)

# Load
eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
print(f"EEG: {eeg.shape}, EOG: {eog.shape}")

# Simple features
def create_features(x, eog_ref):
    feats = [x]
    for low, high in [(1, 30), (3, 25), (5, 20)]:
        b, a = signal.butter(3, [low/128, high/128], btype='band')
        feats.append(signal.filtfilt(b, a, x))
    for alpha in [0.5, 1.0, 1.5]:
        feats.append(x - alpha * eog_ref)
    return np.concatenate(feats)

def mix_at_snr(clean, noise, snr_db):
    k = np.sqrt(np.mean(clean**2) / (10**(snr_db/10) * np.mean(noise**2)))
    return clean + k * noise

# Test at different SNRs
print("\n[Results by SNR]")
print("-"*60)
print(f"{'SNR':>8} {'Pearson':>12} {'RRMSE':>12}")
print("-"*60)

# Use same data for train/test to see theoretical max
train_idx = range(500)
test_idx = range(500, 700)

for snr_db in [-10, -7, -5, -3, 0, 3, 5, 10]:
    # Prepare
    X_tr, y_tr = [], []
    for i in train_idx:
        e = eeg[i]
        o = eog[i % len(eog)]
        noisy = mix_at_snr(e, o, snr_db)
        X_tr.append(create_features(noisy, o))
        y_tr.append(e)
    
    X_tr = np.array(X_tr, dtype=np.float32)
    y_tr = np.array(y_tr, dtype=np.float32)
    X_tr = np.nan_to_num(X_tr, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    
    model = Ridge(alpha=1.0)
    model.fit(X_tr, y_tr)
    
    # Test
    X_te, y_te = [], []
    for i in test_idx:
        e = eeg[i]
        o = eog[i % len(eog)]
        noisy = mix_at_snr(e, o, snr_db)
        X_te.append(create_features(noisy, o))
        y_te.append(e)
    
    X_te = np.nan_to_num(np.array(X_te, dtype=np.float32), nan=0)
    y_te = np.array(y_te)
    X_te = scaler.transform(X_te)
    
    preds = model.predict(X_te)
    
    # Metrics
    corrs = []
    rrmses = []
    for i in range(len(test_idx)):
        c, _ = pearsonr(preds[i], y_te[i])
        mse = np.mean((preds[i] - y_te[i])**2)
        rmse = np.sqrt(mse)
        rrmse = rmse / np.sqrt(np.mean(y_te[i]**2))
        corrs.append(c)
        rrmses.append(rrmse)
    
    mean_c = np.mean(corrs)
    mean_r = np.mean(rrmses)
    
    print(f"{snr_db:>7} dB {mean_c:>12.4f} {mean_r:>12.4f}")

print("-"*60)
print("\nTarget: Pearson >= 0.98, RRMSE <= 0.15")
