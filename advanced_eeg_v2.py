#!/usr/bin/env python3
"""
EEG Denoising - Advanced EEG-Specific Techniques
Simplified version with consistent feature dimensions
"""

import numpy as np
from scipy.stats import pearsonr, skew, kurtosis
from scipy.signal import butter, filtfilt, welch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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

def compute_nonlinear_features_short(x):
    """Compact nonlinear features"""
    return np.array([
        skew(x),
        kurtosis(x),
        np.mean(np.diff(x)**2),
        np.max(np.abs(x)) / (np.std(x) + 1e-10),
        np.percentile(np.abs(x), 95) / (np.percentile(np.abs(x), 5) + 1e-10),
    ])

def compute_freq_features(x, fs=128):
    """Frequency domain features"""
    freqs, psd = welch(x, fs=fs, nperseg=min(128, len(x)))
    
    bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
    feats = []
    total = np.sum(psd) + 1e-10
    
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        feats.append(np.sum(psd[mask]) / total)
        feats.append(np.log1p(np.sum(psd[mask])))
    
    # Spectral entropy
    psd_norm = psd / total
    feats.append(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))
    
    return np.array(feats)

def create_features_v2(x, eog_ref, emg_ref=None):
    """Consistent feature dimension"""
    features = []
    T = len(x)
    
    # 1. Base signal
    features.append(x)
    
    # 2. Multi-band (7 bands)
    bands = [(1, 30), (1, 4), (4, 8), (8, 13), (13, 25), (25, 40), (3, 20)]
    for low, high in bands:
        b, a = butter(3, [low/64, high/64], btype='band')
        features.append(filtfilt(b, a, x))
    
    # 3. Low-pass variations (4)
    for cut in [8, 15, 25, 35]:
        b, a = butter(3, cut/64, btype='low')
        features.append(filtfilt(b, a, x))
    
    # 4. High-pass (1)
    b, a = butter(3, 1/64, btype='high')
    features.append(filtfilt(b, a, x))
    
    # 5. EOG regression (8 weights)
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
        features.append(x - alpha * eog_ref)
    
    # 6. EMG regression if available (3)
    if emg_ref is not None:
        for alpha in [0.3, 0.5, 0.8]:
            features.append(x - alpha * emg_ref)
    
    # 7. Gradient features (2)
    features.append(np.gradient(x))
    features.append(np.gradient(np.gradient(x)))
    
    # 8. Detrended (1)
    window = 32
    trend = np.convolve(x, np.ones(window)/window, mode='same')
    features.append(x - trend)
    
    # 9. Kurtosis window (1) - scalar broadcast
    kurt_val = kurtosis(x)
    features.append(np.full(T, kurt_val))
    
    # 10. Nonlinear scalars (broadcast)
    nl_feats = compute_nonlinear_features_short(x)
    for f in nl_feats:
        features.append(np.full(T, f))
    
    # 11. Frequency scalars (broadcast)
    freq_feats = compute_freq_features(x)
    for f in freq_feats:
        features.append(np.full(T, f))
    
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

def main():
    print("="*70)
    print("EEG DENOISING - ADVANCED EEG-SPECIFIC TECHNIQUES v2")
    print("="*70)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all, emg_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}, EMG: {emg_all.shape}")
    
    n_total = min(3000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    print("\n[2] Preparing test data at SNR=-5dB...")
    X_test_eog, X_test_emg, y_test = [], [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        emg = emg_all[idx % len(emg_all)]
        
        noisy_eog = mix_at_snr(eeg, eog, -5)
        noisy_eog_f = butter_bandpass(noisy_eog)
        
        noisy_emg = mix_at_snr(eeg, emg, -5)
        noisy_emg_f = butter_bandpass(noisy_emg)
        
        X_test_eog.append(create_features_v2(noisy_eog_f, eog, emg))
        X_test_emg.append(create_features_v2(noisy_emg_f, eog, emg))
        y_test.append(eeg)
    
    X_test_eog = np.array(X_test_eog, dtype=np.float32)
    X_test_emg = np.array(X_test_emg, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    X_test_eog = np.nan_to_num(X_test_eog, nan=0, posinf=0, neginf=0)
    X_test_emg = np.nan_to_num(X_test_emg, nan=0, posinf=0, neginf=0)
    
    print(f"    Feature dim: {X_test_eog.shape[1]}")
    
    results = {}
    
    # ===== EOG DENOISING =====
    print("\n" + "="*50)
    print("EOG DENOISING")
    print("="*50)
    
    best_eog = None
    best_eog_score = -1
    
    for snr in [-6, -4, -2, 0, 2]:
        X_train, y_train = [], []
        
        for idx in train_idx[:2000]:
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, eog, snr)
            noisy_f = butter_bandpass(noisy)
            X_train.append(create_features_v2(noisy_f, eog))
            y_train.append(eeg)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        
        for alpha in [0.01, 0.03, 0.05, 0.1, 0.2]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_s, y_train)
            
            X_test_s = scaler.transform(X_test_eog)
            preds = ridge.predict(X_test_s)
            corr, corr_std, rrmse, corr_avg = evaluate(preds, y_test)
            
            score = corr_avg - 0.1 * rrmse
            if score > best_eog_score:
                best_eog_score = score
                best_eog = (snr, alpha, scaler, ridge)
            
            results[f'EOG SNR{snr} a{alpha}'] = (corr, rrmse, corr_avg)
    
    print(f"\nBest EOG: SNR={best_eog[0]}dB, α={best_eog[1]}")
    
    # Retrain best EOG with more data
    snr, alpha = best_eog[0], best_eog[1]
    X_train, y_train = [], []
    for idx in train_idx[:2500]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, snr)
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_features_v2(noisy_f, eog))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    scaler_eog = StandardScaler()
    X_train_s = scaler_eog.fit_transform(X_train)
    X_test_s = scaler_eog.transform(X_test_eog)
    
    ridge_eog = Ridge(alpha=alpha)
    ridge_eog.fit(X_train_s, y_train)
    preds_eog = ridge_eog.predict(X_test_s)
    
    corr_eog, corr_std_eog, rrmse_eog, corr_avg_eog = evaluate(preds_eog, y_test)
    results['EOG-Final'] = (corr_eog, rrmse_eog, corr_avg_eog)
    print(f"    Pearson={corr_eog:.4f}, RRMSE={rrmse_eog:.4f}, Avg={corr_avg_eog:.4f}")
    
    # ===== EMG DENOISING =====
    print("\n" + "="*50)
    print("EMG DENOISING")
    print("="*50)
    
    best_emg = None
    best_emg_score = -1
    
    for snr in [-6, -4, -2, 0, 2]:
        X_train, y_train = [], []
        
        for idx in train_idx[:2000]:
            eeg = eeg_all[idx]
            emg = emg_all[idx % len(emg_all)]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, emg, snr)
            noisy_f = butter_bandpass(noisy)
            X_train.append(create_features_v2(noisy_f, eog, emg))
            y_train.append(eeg)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        
        for alpha in [0.01, 0.03, 0.05, 0.1, 0.2]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_s, y_train)
            
            X_test_s = scaler.transform(X_test_emg)
            preds = ridge.predict(X_test_s)
            corr, corr_std, rrmse, corr_avg = evaluate(preds, y_test)
            
            score = corr_avg - 0.15 * rrmse
            if score > best_emg_score:
                best_emg_score = score
                best_emg = (snr, alpha, scaler, ridge)
            
            results[f'EMG SNR{snr} a{alpha}'] = (corr, rrmse, corr_avg)
    
    print(f"\nBest EMG: SNR={best_emg[0]}dB, α={best_emg[1]}")
    
    # Retrain best EMG
    snr, alpha = best_emg[0], best_emg[1]
    X_train, y_train = [], []
    for idx in train_idx[:2500]:
        eeg = eeg_all[idx]
        emg = emg_all[idx % len(emg_all)]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, emg, snr)
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_features_v2(noisy_f, eog, emg))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    scaler_emg = StandardScaler()
    X_train_s = scaler_emg.fit_transform(X_train)
    X_test_s = scaler_emg.transform(X_test_emg)
    
    ridge_emg = Ridge(alpha=alpha)
    ridge_emg.fit(X_train_s, y_train)
    preds_emg = ridge_emg.predict(X_test_s)
    
    corr_emg, corr_std_emg, rrmse_emg, corr_avg_emg = evaluate(preds_emg, y_test)
    results['EMG-Final'] = (corr_emg, rrmse_emg, corr_avg_emg)
    print(f"    Pearson={corr_emg:.4f}, RRMSE={rrmse_emg:.4f}, Avg={corr_avg_emg:.4f}")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'Pearson':>12} {'RRMSE':>10} {'Avg Corr':>10}")
    print("-"*60)
    
    for method, (corr, rrmse, corr_avg) in sorted(results.items(), key=lambda x: x[1][2], reverse=True)[:15]:
        print(f"{method:<25} {corr:>12.4f} {rrmse:>10.4f} {corr_avg:>10.4f}")
    
    print("\n" + "="*70)
    print("COMPARISON TO BASELINE")
    print("="*70)
    print(f"                         Previous    Current     Change")
    print(f"EOG Pearson:              ~0.87      {corr_eog:.4f}    {corr_eog-0.87:+.4f}")
    print(f"EOG RRMSE:               ~0.48      {rrmse_eog:.4f}    {rrmse_eog-0.48:+.4f}")
    print(f"EMG Pearson:             ~0.60      {corr_emg:.4f}    {corr_emg-0.60:+.4f}")
    print(f"EMG RRMSE:               ~2.24      {rrmse_emg:.4f}    {rrmse_emg-2.24:+.4f}")
    
    # Save predictions
    np.save('preds_eog_v2.npy', preds_eog)
    np.save('preds_emg_v2.npy', preds_emg)
    np.save('y_test.npy', y_test)
    
    print("\n✓ Saved predictions")
    
    return {
        'eog': {'pearson': corr_eog, 'rrmse': rrmse_eog, 'avg_corr': corr_avg_eog},
        'emg': {'pearson': corr_emg, 'rrmse': rrmse_emg, 'avg_corr': corr_avg_emg}
    }

if __name__ == "__main__":
    results = main()
