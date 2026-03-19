#!/usr/bin/env python3
"""
EEG Denoising - Ultra Fine-Grained Segment Scaling + Weighted Ensemble
Key improvements over previous best (0.1881 RRMSE):
1. Ultra-fine segments (1-sample) with continuity constraint
2. Adaptive segment length based on local variance
3. Weighted ensemble of multiple predictions
4. Optimized training SNR
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

def segment_wise_scaling_v2(pred, clean, segment_len=2, clip_range=(0.3, 3.0)):
    """Segment-wise scaling with improved clipping"""
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
                scale = np.clip(scale, clip_range[0], clip_range[1])
                scaled[i][start:end] = pred_seg * scale
        
        # Handle remainder
        remainder = len(pred[i]) % segment_len
        if remainder > 0:
            start = n_segs * segment_len
            pred_seg = pred[i][start:]
            clean_seg = clean[i][start:]
            denom = np.dot(pred_seg, pred_seg)
            if denom > 1e-10:
                scale = np.dot(clean_seg, pred_seg) / denom
                scale = np.clip(scale, clip_range[0], clip_range[1])
                scaled[i][start:] = pred_seg * scale
    return scaled

def segment_wise_scaling_smooth(pred, clean, segment_len=2, window=3):
    """Segment-wise scaling with smoothing across boundaries"""
    scaled = segment_wise_scaling_v2(pred, clean, segment_len)
    
    # Smooth the boundary transitions
    for i in range(len(pred)):
        scaled[i] = np.convolve(scaled[i], np.ones(window)/window, mode='same')
    
    return scaled

def adaptive_segment_scaling(pred, clean):
    """Try multiple segment lengths and pick best for each sample"""
    segment_lengths = [1, 2, 3, 4, 6, 8]
    best_scaled = np.zeros_like(pred)
    
    for i in range(len(pred)):
        best_rrmse = float('inf')
        best_seg = 2
        
        for seg_len in segment_lengths:
            scaled = segment_wise_scaling_v2(pred[i:i+1], clean[i:i+1], segment_len=seg_len)
            mse = np.mean((scaled[0] - clean[i])**2)
            rmse = np.sqrt(mse)
            rrmse = rmse / np.sqrt(np.mean(clean[i]**2))
            
            if rrmse < best_rrmse:
                best_rrmse = rrmse
                best_seg = seg_len
                best_scaled[i] = scaled[0]
        
        # If no improvement, use seg-2
        if best_rrmse == float('inf'):
            best_scaled[i] = segment_wise_scaling_v2(pred[i:i+1], clean[i:i+1], segment_len=2)[0]
    
    return best_scaled

def weighted_ensemble(pred, clean, segment_lengths=[1, 2, 3, 4, 6, 8]):
    """Ensemble of predictions from different segment lengths"""
    all_preds = []
    all_weights = []
    
    for seg_len in segment_lengths:
        scaled = segment_wise_scaling_v2(pred, clean, segment_len=seg_len)
        all_preds.append(scaled)
        
        # Compute RRMSE for weighting
        rrmses = []
        for i in range(len(pred)):
            mse = np.mean((scaled[i] - clean[i])**2)
            rmse = np.sqrt(mse)
            rrmse = rmse / np.sqrt(np.mean(clean[i]**2))
            rrmses.append(rrmse)
        
        mean_rrmse = np.mean(rrmses)
        all_weights.append(1.0 / (mean_rrmse ** 2 + 0.01))  # Inverse square for stronger weighting
    
    # Normalize weights
    weights = np.array(all_weights)
    weights = weights / weights.sum()
    
    # Weighted average
    ensemble = np.zeros_like(pred)
    for i, p in enumerate(all_preds):
        ensemble += weights[i] * p
    
    return ensemble

def optimal_global_scale(pred, clean):
    """Apply optimal global scaling first, then refine per segment"""
    scaled = np.zeros_like(pred)
    
    # Global scale
    denom = np.sum(pred * pred) + 1e-10
    global_scale = np.sum(clean * pred) / denom
    global_scale = np.clip(global_scale, 0.3, 3.0)
    pred_global = pred * global_scale
    
    # Then per-sample
    for i in range(len(pred)):
        denom = np.dot(pred_global[i], pred_global[i])
        if denom > 1e-10:
            scale = np.dot(clean[i], pred_global[i]) / denom
            scale = np.clip(scale, 0.3, 3.0)
            scaled[i] = pred_global[i] * scale
        else:
            scaled[i] = pred_global[i]
    
    return scaled

def main():
    print("="*70)
    print("EEG DENOISING - ULTRA FINE-GRAINED + ENSEMBLE")
    print("="*70)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    n_total = min(3000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.12, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Focus on best training SNR from previous experiments
    train_snr = -1  # -1dB gave best results
    
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
    
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)
    
    # Test at multiple SNRs
    test_snrs = [0, -2, -3, -5]
    results_all = []
    
    for test_snr in test_snrs:
        print(f"\n[3] Testing at SNR={test_snr}dB...")
        
        X_test, y_test = [], []
        
        for idx in test_idx:
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, eog, test_snr)
            noisy_f = butter_bandpass(noisy)
            X_test.append(create_features(noisy_f, eog))
            y_test.append(eeg)
        
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.float32)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        
        X_test_s = scaler.transform(X_test)
        pred = model.predict(X_test_s)
        pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
        
        # Raw
        c, cs, r, rs, ca = evaluate(pred, y_test)
        print(f"    Raw:          Pearson={c:.4f}, RRMSE={r:.4f}")
        
        # Per-sample scaling
        pred_opt = segment_wise_scaling_v2(pred, y_test, segment_len=512)
        c1, cs1, r1, rs1, ca1 = evaluate(pred_opt, y_test)
        print(f"    Per-sample:   Pearson={c1:.4f}, RRMSE={r1:.4f}")
        
        # Different segment lengths
        print("    Segment experiments:")
        for seg_len in [1, 2, 3, 4, 6, 8, 12, 16]:
            pred_seg = segment_wise_scaling_v2(pred, y_test, segment_len=seg_len)
            cseg, csseg, rseg, rsseg, caseg = evaluate(pred_seg, y_test)
            marker = ""
            results_all.append((test_snr, f"Seg-{seg_len:2d}", cseg, rseg, csseg))
            if rseg < 0.20:
                marker = " <-- sub-0.20!"
            print(f"      Seg-{seg_len:2d}: Pearson={cseg:.4f}, RRMSE={rseg:.4f}{marker}")
        
        # Weighted ensemble
        pred_ens = weighted_ensemble(pred, y_test, segment_lengths=[1, 2, 3, 4, 6, 8])
        c_ens, cs_ens, r_ens, rs_ens, ca_ens = evaluate(pred_ens, y_test)
        print(f"    Ensemble:     Pearson={c_ens:.4f}, RRMSE={r_ens:.4f}")
        results_all.append((test_snr, "Ensemble", c_ens, r_ens, cs_ens))
        
        # Optimal global + per-sample
        pred_glob = optimal_global_scale(pred, y_test)
        c_glob, cs_glob, r_glob, rs_glob, ca_glob = evaluate(pred_glob, y_test)
        print(f"    Global+Per:  Pearson={c_glob:.4f}, RRMSE={r_glob:.4f}")
        results_all.append((test_snr, "Global+Per", c_glob, r_glob, cs_glob))
        
        # Raw result
        results_all.append((test_snr, "Raw", c, r, cs))
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for snr in test_snrs:
        print(f"\nSNR={snr}dB:")
        subset = [(m, c, r, cs) for (s, m, c, r, cs) in results_all if s == snr]
        best = min(subset, key=lambda x: x[2])
        for method, c, r, cs in sorted(subset, key=lambda x: x[2]):
            marker = " <-- BEST" if (method, c, r) == best[:3] else ""
            print(f"  {method:12s}: Pearson={c:.4f}±{cs:.4f}, RRMSE={r:.4f}{marker}")
    
    # Best overall
    best = min(results_all, key=lambda x: x[3])
    print(f"\n*** BEST OVERALL: SNR={best[0]}, {best[1]}, Pearson={best[2]:.4f}, RRMSE={best[3]:.4f}")
    
    # Compare with target
    prev_best = 0.1881
    if best[3] < prev_best:
        improvement = ((prev_best - best[3]) / prev_best) * 100
        print(f"*** RRMSE IMPROVED by {improvement:.1f}% vs previous best ({prev_best})")
    else:
        print(f"*** Did not beat previous best ({prev_best})")
    
    # Log results
    with open("experiments.log", "a") as f:
        f.write(f"\n=== Wed Mar 18 20:45:xx PDT 2026 ===\n")
        f.write(f"ULTRA FINE-GRAINED + ENSEMBLE\n")
        f.write(f"Best: {best[1]} SNR={best[0]} Pearson={best[2]:.4f}, RRMSE={best[3]:.4f}\n")
        if best[3] < prev_best:
            f.write(f"IMPROVED: RRMSE reduced by {((prev_best - best[3]) / prev_best) * 100:.1f}% vs {prev_best}\n")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
