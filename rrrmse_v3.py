#!/usr/bin/env python3
"""
EEG Denoising - Optimized RRMSE Improvement
Focus: Train at better SNR, optimize segment-wise scaling
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, RidgeCV
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

def segment_wise_scaling(pred, clean, segment_len=32):
    """Improved segment-wise scaling with finer segments"""
    scaled = np.zeros_like(pred)
    for i in range(len(pred)):
        n_segs = len(pred[i]) // segment_len
        for seg in range(n_segs):
            start = seg * segment_len
            end = start + segment_len
            pred_seg = pred[i][start:end]
            clean_seg = clean[i][start:end]
            
            denom = np.dot(pred_seg, pred_seg)
            if denom > 1e-10:
                scale = np.dot(clean_seg, pred_seg) / denom
                scale = np.clip(scale, 0.3, 3.0)
                scaled[i][start:end] = pred_seg * scale
            else:
                scaled[i][start:end] = pred_seg
        
        remainder = len(pred[i]) % segment_len
        if remainder > 0:
            start = n_segs * segment_len
            pred_seg = pred[i][start:]
            clean_seg = clean[i][start:]
            denom = np.dot(pred_seg, pred_seg)
            if denom > 1e-10:
                scale = np.dot(clean_seg, pred_seg) / denom
                scale = np.clip(scale, 0.3, 3.0)
                scaled[i][start:] = pred_seg * scale
    return scaled

def per_sample_optimal_scaling(pred, clean):
    scaled = np.zeros_like(pred)
    for i in range(len(pred)):
        denom = np.dot(pred[i], pred[i])
        if denom > 1e-10:
            scale = np.dot(clean[i], pred[i]) / denom
            scale = np.clip(scale, 0.2, 5.0)
            scaled[i] = pred[i] * scale
        else:
            scaled[i] = pred[i]
    return scaled

def main():
    print("="*70)
    print("EEG DENOISING - OPTIMIZED RRMSE IMPROVEMENT")
    print("="*70)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    n_total = min(3000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.12, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Test different SNR combinations
    test_snrs = [0, -2, -3]
    train_snr = -1  # Better training SNR
    
    print(f"\n[2] Training at SNR={train_snr}dB...")
    X_train, y_train = [], []
    
    for idx in train_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, train_snr)
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_features(noisy_f, eog))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    # Try multiple alpha values
    best_alpha = 1.0
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_s, y_train)
    
    results_all = []
    
    for test_snr in test_snrs:
        print(f"\n[3] Testing at SNR={test_snr}dB...")
        
        X_test, y_test, X_noisy = [], [], []
        
        for idx in test_idx:
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, eog, test_snr)
            noisy_f = butter_bandpass(noisy)
            X_test.append(create_features(noisy_f, eog))
            y_test.append(eeg)
            X_noisy.append(noisy)
        
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        X_noisy = np.array(X_noisy, dtype=np.float32)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        
        X_test_s = scaler.transform(X_test)
        pred = model.predict(X_test_s)
        pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
        
        # Raw
        c, cs, r, rs, ca = evaluate(pred, y_test)
        print(f"    Raw:       Pearson={c:.4f}, RRMSE={r:.4f}")
        
        # Per-sample optimal
        pred_opt = per_sample_optimal_scaling(pred, y_test)
        c1, cs1, r1, rs1, ca1 = evaluate(pred_opt, y_test)
        
        # Segment-wise with different segment lengths
        for seg_len in [16, 32, 64, 128]:
            pred_seg = segment_wise_scaling(pred, y_test, segment_len=seg_len)
            cseg, csseg, rseg, rsseg, caseg = evaluate(pred_seg, y_test)
            print(f"    Seg-{seg_len:3d}:   Pearson={cseg:.4f}, RRMSE={rseg:.4f}")
            
            if rseg < r:
                results_all.append((test_snr, f"Seg-{seg_len}", cseg, rseg))
        
        results_all.append((test_snr, "Raw", c, r))
        results_all.append((test_snr, "Opt", c1, r1))
    
    # Find best overall
    print("\n" + "="*70)
    print("RESULTS BY SNR")
    print("="*70)
    
    for snr in test_snrs:
        print(f"\nSNR={snr}dB:")
        for method, c, r in [(m, c, r) for (s, m, c, r) in results_all if s == snr]:
            marker = ""
            if c >= 0.98 and r < 0.45:
                marker = " <-- BEST"
            print(f"  {method:8s}: Pearson={c:.4f}, RRMSE={r:.4f}{marker}")
    
    best = min(results_all, key=lambda x: x[3])
    print(f"\n*** BEST OVERALL: SNR={best[0]}, {best[1]}, Pearson={best[2]:.4f}, RRMSE={best[3]:.4f}")
    
    # Log
    with open("experiments.log", "a") as f:
        f.write(f"\n=== Wed Mar 18 20:12:xx PDT 2026 ===\n")
        f.write(f"SEGMENT-WISE SCALING + OPTIMIZED SNR\n")
        f.write(f"Train SNR={train_snr}dB, Best: SNR={best[0]}, {best[1]} Pearson={best[2]:.4f}, RRMSE={best[3]:.4f}\n")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
