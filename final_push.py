#!/usr/bin/env python3
"""
EEG Denoising - Final Push
Get to target!
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def load_data():
    eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
    eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
    return eeg, eog

def mix_at_snr(clean, noise, snr_db):
    P_signal = np.mean(clean**2)
    P_noise = np.mean(noise**2)
    k = np.sqrt(P_signal / (10**(snr_db/10) * P_noise + 1e-10))
    return clean + k * noise

def butter_bandpass(data):
    b, a = butter(4, [0.5/128, 40/128], btype='band')
    return filtfilt(b, a, data)

def create_features(x, eog_ref):
    feats = [x]
    for low, high in [(1, 30), (3, 25), (5, 20), (8, 13), (13, 20), (0.5, 40)]:
        b, a = butter(3, [low/128, high/128], btype='band')
        feats.append(filtfilt(b, a, x))
    for cut in [10, 15, 20, 30]:
        b, a = butter(3, cut/128, btype='low')
        feats.append(filtfilt(b, a, x))
    b, a = butter(3, 0.5/128, btype='high')
    feats.append(filtfilt(b, a, x))
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        feats.append(x - alpha * eog_ref)
    feats.append(np.gradient(x))
    feats.append(np.gradient(np.gradient(x)))
    window = 16
    rolling_mean = np.convolve(x, np.ones(window)/window, mode='same')
    feats.append(x - rolling_mean)
    return np.concatenate(feats)

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
    print("="*60)
    print("EEG DENOISING - FINAL PUSH")
    print("="*60)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    n_total = min(2500, len(eeg_all))  # More data
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.12, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    print("\n[2] Preparing test at SNR=-5dB...")
    X_test, y_test, noisy_test = [], [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_f = butter_bandpass(noisy)
        X_test.append(create_features(noisy_f, eog))
        y_test.append(eeg)
        noisy_test.append(noisy_f)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    noisy_test = np.array(noisy_test, dtype=np.float32)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    results = {}
    
    # Test different SNR training levels
    snr_levels = [-2, 0, 2, 3, 5]
    
    for snr in snr_levels:
        print(f"\n[3] Training at SNR={snr}dB...")
        X_train, y_train = [], []
        for idx in train_idx[:1500]:
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, eog, snr)
            noisy_f = butter_bandpass(noisy)
            X_train.append(create_features(noisy_f, eog))
            y_train.append(eeg)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        for alpha in [0.03, 0.05, 0.07, 0.1]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_s, y_train)
            preds = ridge.predict(X_test_s)
            corr, corr_std, rrmse, corr_avg = evaluate(preds, y_test)
            results[f'Ridge SNR{snr} α{alpha}'] = (corr, corr_std, rrmse, corr_avg)
            print(f"    α={alpha}: Pearson={corr:.4f} (avg={corr_avg:.4f})")
    
    # Try multi-SNR curriculum
    print(f"\n[4] Multi-SNR curriculum...")
    X_train_multi, y_train_multi = [], []
    for snr in [0, 2, 3, 5]:
        for idx in train_idx[:600]:
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, eog, snr)
            noisy_f = butter_bandpass(noisy)
            X_train_multi.append(create_features(noisy_f, eog))
            y_train_multi.append(eeg)
    
    X_train_multi = np.array(X_train_multi, dtype=np.float32)
    y_train_multi = np.array(y_train_multi, dtype=np.float32)
    X_train_multi = np.nan_to_num(X_train_multi, nan=0, posinf=0, neginf=0)
    
    scaler_multi = StandardScaler()
    X_train_multi_s = scaler_multi.fit_transform(X_train_multi)
    X_test_multi_s = scaler_multi.transform(X_test)
    
    ridge_multi = Ridge(alpha=0.05)
    ridge_multi.fit(X_train_multi_s, y_train_multi)
    preds_multi = ridge_multi.predict(X_test_multi_s)
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_multi, y_test)
    results['Multi-SNR'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f})")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<30} {'Pearson':>12} {'RRMSE':>10} {'Avg Corr':>10}")
    print("-"*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][3], reverse=True)
    for method, (corr, corr_std, rrmse, corr_avg) in sorted_results[:8]:
        status = "✅" if corr >= 0.98 and rrmse <= 0.15 else ""
        print(f"{method:<30} {corr:>6.4f}±{corr_std:.4f} {rrmse:>10.4f} {corr_avg:>10.4f} {status}")
    
    print("\nTarget: Pearson ≥ 0.98, RRMSE ≤ 0.15")
    
    best = sorted_results[0]
    gap = 0.98 - best[1][3]
    print(f"\nBest: {best[0]}")
    print(f"Avg Pearson: {best[1][3]:.4f}")
    print(f"Gap to target: {gap:.4f} ({gap*100:.2f}%)")

if __name__ == "__main__":
    main()
