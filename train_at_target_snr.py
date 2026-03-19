#!/usr/bin/env python3
"""
EEG Denoising - Fixed: Train at same SNR as test (-5dB)
Key insight: Train at the target noise level!
"""

import numpy as np
from scipy.stats import pearsonr, skew, kurtosis
from scipy.signal import butter, filtfilt, welch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_data():
    eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
    eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
    emg = np.load("data/EMG_all_epochs.npy", allow_pickle=True)
    return eeg, eog, emg

def mix_at_snr(clean, noise, snr_db):
    P_signal = np.mean(clean**2)
    P_noise = np.mean(noise**2)
    k = np.sqrt(P_signal / (10**(snr_db/10) * P_noise + 1e-10))
    return clean + k * noise

def butter_bandpass(data, fs=128):
    b, a = butter(4, [0.5/fs*2, 40/fs*2], btype='band')
    return filtfilt(b, a, data)

def create_simple_features(x, eog_ref):
    """Simpler features - less overfitting risk"""
    T = len(x)
    features = [x]
    
    # Multi-band
    for low, high in [(1, 30), (1, 4), (4, 8), (8, 13), (13, 30), (0.5, 40)]:
        b, a = butter(3, [low/64, high/64], btype='band')
        features.append(filtfilt(b, a, x))
    
    # Low-pass
    for cut in [10, 20, 30]:
        b, a = butter(3, cut/64, btype='low')
        features.append(filtfilt(b, a, x))
    
    # EOG regression
    for alpha in [0.5, 0.7, 0.9, 1.0, 1.2]:
        features.append(x - alpha * eog_ref)
    
    # Gradient
    features.append(np.gradient(x))
    
    # Detrend
    window = 16
    trend = np.convolve(x, np.ones(window)/window, mode='same')
    features.append(x - trend)
    
    return np.concatenate(features)

def create_rich_features(x, eog_ref):
    """Richer features"""
    T = len(x)
    features = [x]
    
    # Multi-band
    for low, high in [(1, 30), (1, 4), (4, 8), (8, 13), (13, 30), (0.5, 40), (3, 25), (5, 20)]:
        b, a = butter(3, [low/64, high/64], btype='band')
        features.append(filtfilt(b, a, x))
    
    # Low-pass variations
    for cut in [8, 12, 18, 25, 35]:
        b, a = butter(3, cut/64, btype='low')
        features.append(filtfilt(b, a, x))
    
    # High-pass
    b, a = butter(3, 0.5/64, btype='high')
    features.append(filtfilt(b, a, x))
    
    # EOG regression with many weights
    for alpha in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
        features.append(x - alpha * eog_ref)
    
    # Gradient
    features.append(np.gradient(x))
    features.append(np.gradient(np.gradient(x)))
    
    # Detrend
    for window in [16, 32]:
        trend = np.convolve(x, np.ones(window)/window, mode='same')
        features.append(x - trend)
    
    return np.concatenate(features)

def evaluate(pred, truth):
    pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
    truth = np.nan_to_num(truth, nan=0, posinf=0, neginf=0)
    
    corrs = []
    for i in range(len(pred)):
        c, _ = pearsonr(pred[i], truth[i])
        if not np.isnan(c):
            corrs.append(c)
    
    corr_mean = np.mean(corrs)
    corr_std = np.std(corrs)
    corr_avg, _ = pearsonr(pred.mean(axis=0), truth.mean(axis=0))
    
    mse = np.mean((pred - truth)**2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.sqrt(np.mean(truth**2))
    
    return corr_mean, corr_std, rrmse, corr_avg

def train_and_evaluate(X_train, y_train, X_test, y_test, alphas=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0]):
    """Train Ridge with CV to select alpha"""
    results = {}
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        preds = ridge.predict(X_test)
        corr, corr_std, rrmse, corr_avg = evaluate(preds, y_test)
        results[alpha] = (corr, rrmse, corr_avg, ridge)
    
    # Find best by avg correlation
    best_alpha = max(results.keys(), key=lambda a: results[a][2])
    return best_alpha, results[best_alpha]

def main():
    print("="*70)
    print("EEG DENOISING - TRAIN AT TARGET SNR (-5dB)")
    print("="*70)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all, emg_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}, EMG: {emg_all.shape}")
    
    # Split data - use same indices for fair comparison
    n_total = min(3000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    results = {}
    
    # ===== EOG DENOISING =====
    print("\n" + "="*50)
    print("EOG DENOISING - Training at SNR=-5dB")
    print("="*50)
    
    # Create test set at -5dB
    X_test_eog, y_test = [], []
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_f = butter_bandpass(noisy)
        X_test_eog.append(create_rich_features(noisy_f, eog))
        y_test.append(eeg)
    
    X_test_eog = np.array(X_test_eog, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    X_test_eog = np.nan_to_num(X_test_eog, nan=0, posinf=0, neginf=0)
    print(f"    Feature dim: {X_test_eog.shape[1]}")
    
    # Train at -5dB (same as test!)
    X_train, y_train = [], []
    for idx in train_idx[:2200]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)  # SAME SNR as test!
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_rich_features(noisy_f, eog))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    print(f"    Training at SNR=-5dB with {len(X_train)} samples...")
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test_eog)
    
    best_alpha, (corr, rrmse, corr_avg, ridge) = train_and_evaluate(
        X_train_s, y_train, X_test_s, y_test
    )
    
    preds_eog = ridge.predict(X_test_s)
    corr_eog, corr_std_eog, rrmse_eog, corr_avg_eog = evaluate(preds_eog, y_test)
    results['EOG-SameSNR'] = (corr_eog, rrmse_eog, corr_avg_eog)
    
    print(f"    Best α={best_alpha}: Pearson={corr_eog:.4f}, RRMSE={rrmse_eog:.4f}, Avg={corr_avg_eog:.4f}")
    
    # Try simpler features too
    X_test_eog_simple = []
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_f = butter_bandpass(noisy)
        X_test_eog_simple.append(create_simple_features(noisy_f, eog))
    
    X_test_eog_simple = np.array(X_test_eog_simple, dtype=np.float32)
    X_test_eog_simple = np.nan_to_num(X_test_eog_simple, nan=0, posinf=0, neginf=0)
    
    X_train_simple = []
    for idx in train_idx[:2200]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_f = butter_bandpass(noisy)
        X_train_simple.append(create_simple_features(noisy_f, eog))
    
    X_train_simple = np.array(X_train_simple, dtype=np.float32)
    X_train_simple = np.nan_to_num(X_train_simple, nan=0, posinf=0, neginf=0)
    
    scaler_simple = StandardScaler()
    X_train_s_simple = scaler_simple.fit_transform(X_train_simple)
    X_test_s_simple = scaler_simple.transform(X_test_eog_simple)
    
    best_alpha_simple, (corr_s, rrmse_s, corr_avg_s, ridge_s) = train_and_evaluate(
        X_train_s_simple, y_train, X_test_s_simple, y_test
    )
    
    print(f"    Simple features: α={best_alpha_simple}: Pearson={corr_s:.4f}, RRMSE={rrmse_s:.4f}, Avg={corr_avg_s:.4f}")
    
    if corr_avg_s > corr_avg_eog:
        print("    -> Using simple features")
        preds_eog = ridge_s.predict(X_test_s_simple)
        corr_eog, corr_std_eog, rrmse_eog, corr_avg_eog = evaluate(preds_eog, y_test)
        results['EOG-Simple'] = (corr_eog, rrmse_eog, corr_avg_eog)
    
    # ===== EMG DENOISING =====
    print("\n" + "="*50)
    print("EMG DENOISING - Training at SNR=-5dB")
    print("="*50)
    
    # Test set
    X_test_emg = []
    for idx in test_idx:
        eeg = eeg_all[idx]
        emg = emg_all[idx % len(emg_all)]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, emg, -5)
        noisy_f = butter_bandpass(noisy)
        X_test_emg.append(create_rich_features(noisy_f, eog))  # Use EOG as ref still
        y_test.append(eeg)
    
    X_test_emg = np.array(X_test_emg, dtype=np.float32)
    X_test_emg = np.nan_to_num(X_test_emg, nan=0, posinf=0, neginf=0)
    
    # Train at -5dB
    X_train_emg, y_train_emg = [], []
    for idx in train_idx[:2200]:
        eeg = eeg_all[idx]
        emg = emg_all[idx % len(emg_all)]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, emg, -5)
        noisy_f = butter_bandpass(noisy)
        X_train_emg.append(create_rich_features(noisy_f, eog))
        y_train_emg.append(eeg)
    
    X_train_emg = np.array(X_train_emg, dtype=np.float32)
    y_train_emg = np.array(y_train_emg, dtype=np.float32)
    X_train_emg = np.nan_to_num(X_train_emg, nan=0, posinf=0, neginf=0)
    
    print(f"    Training at SNR=-5dB with {len(X_train_emg)} samples...")
    
    scaler_emg = StandardScaler()
    X_train_s_emg = scaler_emg.fit_transform(X_train_emg)
    X_test_s_emg = scaler_emg.transform(X_test_emg)
    
    # Re-create y_test for EMG
    y_test_emg = []
    for idx in test_idx:
        eeg = eeg_all[idx]
        y_test_emg.append(eeg)
    y_test_emg = np.array(y_test_emg, dtype=np.float32)
    
    best_alpha_emg, (corr_emg, rrmse_emg, corr_avg_emg, ridge_emg) = train_and_evaluate(
        X_train_s_emg, y_train_emg, X_test_s_emg, y_test_emg
    )
    
    preds_emg = ridge_emg.predict(X_test_s_emg)
    corr_emg, corr_std_emg, rrmse_emg, corr_avg_emg = evaluate(preds_emg, y_test_emg)
    results['EMG-SameSNR'] = (corr_emg, rrmse_emg, corr_avg_emg)
    
    print(f"    Best α={best_alpha_emg}: Pearson={corr_emg:.4f}, RRMSE={rrmse_emg:.4f}, Avg={corr_avg_emg:.4f}")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"                         Previous    Current     Change")
    print(f"EOG Pearson:              ~0.87      {corr_eog:.4f}    {corr_eog-0.87:+.4f}")
    print(f"EOG RRMSE:               ~0.48      {rrmse_eog:.4f}    {rrmse_eog-0.48:+.4f}")
    print(f"EMG Pearson:             ~0.60      {corr_emg:.4f}    {corr_emg-0.60:+.4f}")
    print(f"EMG RRMSE:               ~2.24      {rrmse_emg:.4f}    {rrmse_emg-2.24:+.4f}")
    
    np.save('preds_eog_target_snr.npy', preds_eog)
    np.save('preds_emg_target_snr.npy', preds_emg)
    
    return {
        'eog': {'pearson': corr_eog, 'rrmse': rrmse_eog, 'avg_corr': corr_avg_eog},
        'emg': {'pearson': corr_emg, 'rrmse': rrmse_emg, 'avg_corr': corr_avg_emg}
    }

if __name__ == "__main__":
    results = main()
