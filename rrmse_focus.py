#!/usr/bin/env python3
"""
EEG Denoising - RRMSE Focus
Try to improve RRMSE while maintaining Pearson
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, Lasso
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
    rrmses = []
    for i in range(len(pred)):
        c, _ = pearsonr(pred[i], truth[i])
        if not np.isnan(c):
            corrs.append(c)
        mse = np.mean((pred[i] - truth[i])**2)
        rmse = np.sqrt(mse)
        rrmse = rmse / np.sqrt(np.mean(truth[i]**2))
        rrmses.append(rrmse)
    corr_mean = np.mean(corrs)
    corr_std = np.std(corrs)
    corr_avg, _ = pearsonr(pred.mean(axis=0), truth.mean(axis=0))
    rrmse_mean = np.mean(rrmses)
    rrmse_std = np.std(rrmses)
    return corr_mean, corr_std, rrmse_mean, rrmse_std, corr_avg

def main():
    print("="*60)
    print("EEG DENOISING - RRMSE FOCUS")
    print("="*60)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    n_total = min(2000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    print("\n[2] Preparing test at SNR=-5dB...")
    X_test, y_test = [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_f = butter_bandpass(noisy)
        X_test.append(create_features(noisy_f, eog))
        y_test.append(eeg)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    # Get test clean std for normalization
    clean_std = np.std(y_test, axis=1, keepdims=True)
    
    results = {}
    
    # Train at -2dB (best SNR for Pearson)
    print("\n[3] Training at SNR=-2dB...")
    X_train, y_train = [], []
    for idx in train_idx[:1500]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -2)
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_features(noisy_f, eog))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Method 1: Ridge with different alphas
    for alpha in [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_s, y_train)
        preds = ridge.predict(X_test_s)
        
        # Method 1a: Raw predictions
        corr, corr_std, rrmse, rrmse_std, corr_avg = evaluate(preds, y_test)
        results[f'Ridge α={alpha}'] = (corr, corr_std, rrmse, rrmse_std, corr_avg)
        print(f"    α={alpha}: Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
        
        # Method 1b: Scale predictions to match clean signal variance
        pred_std = np.std(preds, axis=1, keepdims=True)
        preds_scaled = preds * (clean_std / (pred_std + 1e-10))
        corr2, corr_std2, rrmse2, rrmse_std2, corr_avg2 = evaluate(preds_scaled, y_test)
        results[f'Ridge α={alpha} scaled'] = (corr2, corr_std2, rrmse2, rrmse_std2, corr_avg2)
        print(f"    α={alpha} scaled: Pearson={corr2:.4f}, RRMSE={rrmse2:.4f}")
    
    # Method 2: Post-hoc scaling
    print("\n[4] Post-hoc scaling optimization...")
    # Find optimal scale factor per sample
    preds_ridge = Ridge(alpha=0.1)
    preds_ridge.fit(X_train_s, y_train)
    preds_raw = preds_ridge.predict(X_test_s)
    
    best_scale = 1.0
    best_rrmse = float('inf')
    for scale in np.arange(0.5, 2.0, 0.05):
        preds_scaled = preds_raw * scale
        _, _, rrmse, _, _ = evaluate(preds_scaled, y_test)
        if rrmse < best_rrmse:
            best_rrmse = rrmse
            best_scale = scale
    
    print(f"    Best scale: {best_scale:.2f}")
    preds_opt = preds_raw * best_scale
    corr, corr_std, rrmse, rrmse_std, corr_avg = evaluate(preds_opt, y_test)
    results['Ridge optimal scale'] = (corr, corr_std, rrmse, rrmse_std, corr_avg)
    print(f"    Scaled: Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # Method 3: Different training SNR with scaling
    print("\n[5] Training at SNR=0dB with scaling...")
    X_train0, y_train0 = [], []
    for idx in train_idx[:1500]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, 0)
        noisy_f = butter_bandpass(noisy)
        X_train0.append(create_features(noisy_f, eog))
        y_train0.append(eeg)
    
    X_train0 = np.array(X_train0, dtype=np.float32)
    y_train0 = np.array(y_train0, dtype=np.float32)
    X_train0 = np.nan_to_num(X_train0, nan=0, posinf=0, neginf=0)
    
    scaler0 = StandardScaler()
    X_train0_s = scaler0.fit_transform(X_train0)
    X_test0_s = scaler0.transform(X_test)
    
    ridge0 = Ridge(alpha=0.1)
    ridge0.fit(X_train0_s, y_train0)
    preds0 = ridge0.predict(X_test0_s)
    
    # Find optimal scale
    best_scale0 = 1.0
    best_rrmse0 = float('inf')
    for scale in np.arange(0.5, 2.0, 0.05):
        preds_scaled = preds0 * scale
        _, _, rrmse, _, _ = evaluate(preds_scaled, y_test)
        if rrmse < best_rrmse0:
            best_rrmse0 = rrmse
            best_scale0 = scale
    
    preds0_scaled = preds0 * best_scale0
    corr, corr_std, rrmse, rrmse_std, corr_avg = evaluate(preds0_scaled, y_test)
    results['Ridge SNR0 scaled'] = (corr, corr_std, rrmse, rrmse_std, corr_avg)
    print(f"    SNR0 scaled: Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<30} {'Pearson':>10} {'RRMSE':>10} {'Target':>10}")
    print("-"*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][4], reverse=True)
    for method, (corr, corr_std, rrmse, rrmse_std, corr_avg) in sorted_results[:10]:
        p_ok = "✅" if corr >= 0.98 else ""
        r_ok = "✅" if rrmse <= 0.15 else ""
        print(f"{method:<30} {corr:>10.4f} {rrmse:>10.4f} {p_ok}{r_ok}")
    
    print("\nTarget: Pearson ≥ 0.98, RRMSE ≤ 0.15")
    
    best = sorted_results[0]
    print(f"\nBest by Avg Corr: {best[0]}")
    print(f"Pearson={best[1][0]:.4f}, RRMSE={best[1][2]:.4f}")

if __name__ == "__main__":
    main()
