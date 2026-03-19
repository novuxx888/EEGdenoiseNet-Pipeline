#!/usr/bin/env python3
"""
EEG Denoising - Advanced Methods with Wavelets and Multi-SNR Training
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import pywt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("EEG DENOISING - ADVANCED METHODS")
print("="*60)

# Load data
print("\n[1] Loading data...")
eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
print(f"    EEG: {eeg.shape}, EOG: {eog.shape}")

# Split
n_total = 2000
indices = np.arange(n_total)
np.random.shuffle(indices)
train_idx = indices[:1700]
test_idx = indices[1700:]

# Wavelet denoising
def wavelet_denoise(signal_data, wavelet='db4', level=4):
    """Wavelet-based denoising"""
    # Decompose
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    # Thresholding (soft thresholding)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal_data)))
    
    # Apply threshold to detail coefficients
    new_coeffs = [coeffs[0]]  # Keep approximation
    for c in coeffs[1:]:
        new_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
    
    # Reconstruct
    return pywt.waverec(new_coeffs, wavelet)[:len(signal_data)]

# Enhanced features with wavelets
def create_enhanced_features(x, eog_ref):
    """Feature engineering with wavelets"""
    feats = [x]
    
    # Bandpass filters
    for low, high in [(1, 30), (3, 25), (5, 20), (8, 15), (1, 40)]:
        b, a = butter(3, [low/128, high/128], btype='band')
        feats.append(filtfilt(b, a, x))
    
    # Lowpass
    for cut in [8, 12, 18, 25]:
        b, a = butter(3, cut/128, btype='low')
        feats.append(filtfilt(b, a, x))
    
    # Wavelet denoised versions
    for wavelet in ['db4', 'sym4', 'coif3']:
        try:
            wd = wavelet_denoise(x, wavelet)
            feats.append(wd)
        except:
            pass
    
    # EOG subtractions (more granular)
    for alpha in np.linspace(0.3, 2.0, 10):
        feats.append(x - alpha * eog_ref)
    
    # Temporal features
    feats.append(np.gradient(x))
    feats.append(signal.medfilt(x, kernel_size=3))
    feats.append(signal.medfilt(x, kernel_size=5))
    
    return np.concatenate(feats)

def mix_at_snr(clean, noise, snr_db):
    k = np.sqrt(np.mean(clean**2) / (10**(snr_db/10) * np.mean(noise**2)))
    return clean + k * noise

# Multi-SNR training
print("\n[2] Training at multiple SNR levels...")

X_train, y_train = [], []

for idx in train_idx:
    e = eeg[idx]
    o = eog[idx % len(eog)]
    
    # Train at multiple SNRs for better generalization
    for snr in [-10, -7, -5, -3, 0]:
        noisy = mix_at_snr(e, o, snr)
        feat = create_enhanced_features(noisy, o)
        X_train.append(feat)
        y_train.append(e)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)

print(f"    Training samples: {X_train.shape}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Test at -5dB
print("\n[3] Testing at SNR=-5dB...")

X_test, y_test = [], []

for idx in test_idx:
    e = eeg[idx]
    o = eog[idx % len(eog)]
    noisy = mix_at_snr(e, o, -5)
    feat = create_enhanced_features(noisy, o)
    X_test.append(feat)
    y_test.append(e)

X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)
X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
X_test_scaled = scaler.transform(X_test)

print(f"    Test samples: {X_test.shape}")

# Train Ridge with CV
print("\n[4] Training Ridge with cross-validation...")
model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10, 100])
model.fit(X_train_scaled, y_train)
print(f"    Best alpha: {model.alpha_}")

# Predict
preds = model.predict(X_test_scaled)

# Evaluate per sample
pearson_scores, rrmse_scores = [], []

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

# Also test classical wavelets directly
print("\n[5] Testing wavelet denoising directly...")

wavelet_preds = []
for i in range(len(test_idx)):
    noisy = mix_at_snr(y_test[i], eog[test_idx[i] % len(eog)], -5)
    wd = wavelet_denoise(noisy)
    wavelet_preds.append(wd)
wavelet_preds = np.array(wavelet_preds)

wavelet_corrs, w_rrmses = [], []
for i in range(len(test_idx)):
    c, _ = pearsonr(wavelet_preds[i], y_test[i])
    mse = np.mean((wavelet_preds[i] - y_test[i])**2)
    rrmse = np.sqrt(mse) / np.sqrt(np.mean(y_test[i]**2))
    wavelet_corrs.append(c)
    w_rrmses.append(rrmse)

mean_wc = np.mean(wavelet_corrs)
mean_wr = np.mean(w_rrmses)

# Results
print("\n" + "="*60)
print("📊 ADVANCED RESULTS")
print("="*60)
print(f"Test samples: {len(test_idx)}")
print(f"Training: Multi-SNR ({1700*5} samples)")
print("-"*60)
print(f"🎯 TARGET: Pearson ≥ 0.98, RRMSE ≤ 0.15")
print("-"*60)
print(f"Wavelet only:     Pearson={mean_wc:.4f}, RRMSE={mean_wr:.4f}")
print(f"Ridge (multi-SNR): Pearson={mean_p:.4f}±{std_p:.4f}, RRMSE={mean_r:.4f}±{std_r:.4f}")
print("-"*60)

if mean_p >= 0.98 and mean_r <= 0.15:
    print("\n🎉 TARGETS ACHIEVED!")
elif mean_p >= 0.9:
    print("\n✅ Good progress!")

print("="*60)
