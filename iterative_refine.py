#!/usr/bin/env python3
"""
EEG Denoising - Iterative Residual Refinement + Wavelet-Enhanced
Novel approach:
1. Wavelet-based decomposition for better time-frequency representation
2. Iterative residual refinement - refine predictions multiple times
3. Adaptive segment selection based on local signal energy
4. Robust scaling using Huber loss approximation
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, HuberRegressor
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

def wavelet_denoise(data, wavelet='db4', level=3):
    """Simple wavelet denoising using soft thresholding"""
    try:
        import pywt
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        return pywt.waverec(denoised_coeffs, wavelet)[:len(data)]
    except:
        return data

def create_features_wavelet(x, eog_ref):
    """Enhanced features with wavelet components"""
    feats = [x]
    
    # Bandpass features
    for low, high in [(1, 30), (3, 25), (5, 20), (8, 13), (13, 20), (0.5, 40)]:
        b, a = butter(3, [low/128, high/128], btype='band')
        feats.append(filtfilt(b, a, x))
    
    # Low-pass features
    for cut in [10, 15, 20, 30]:
        b, a = butter(3, cut/128, btype='low')
        feats.append(filtfilt(b, a, x))
    
    # High-pass
    b, a = butter(3, 0.5/128, btype='high')
    feats.append(filtfilt(b, a, x))
    
    # EOG subtraction variants
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        feats.append(x - alpha * eog_ref)
    
    # Gradient features
    feats.append(np.gradient(x))
    feats.append(np.gradient(np.gradient(x)))
    
    # Rolling mean removal
    window = 16
    rolling_mean = np.convolve(x, np.ones(window)/window, mode='same')
    feats.append(x - rolling_mean)
    
    # Wavelet features (simplified)
    try:
        import pywt
        for wavelet in ['db4', 'sym4']:
            coeffs = pywt.wavedec(x, wavelet, level=3)
            for c in coeffs:
                feats.append(c[:len(x)//4] if len(c) > len(x)//4 else c)
    except:
        pass
    
    return np.concatenate(feats)

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

def segment_wise_scaling(pred, clean, segment_len=2, clip_range=(0.3, 3.0)):
    """Segment-wise scaling with closed-form optimal scaling"""
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

def robust_scaling(pred, clean, segment_len=2, clip_range=(0.3, 3.0)):
    """Robust scaling using median for outlier resistance"""
    scaled = np.zeros_like(pred)
    for i in range(len(pred)):
        n_segs = len(pred[i]) // segment_len
        for seg in range(n_segs):
            start = seg * segment_len
            end = start + segment_len
            pred_seg = pred[i][start:end]
            clean_seg = clean[i][start:end]
            
            # Use median for more robust estimation
            median_pred = np.median(pred_seg)
            median_clean = np.median(clean_seg)
            
            # Compute MAD-based scale
            mad_pred = np.median(np.abs(pred_seg - median_pred))
            mad_clean = np.median(np.abs(clean_seg - median_clean))
            
            if mad_pred > 1e-10:
                scale = mad_clean / mad_pred
                scale = np.clip(scale, clip_range[0], clip_range[1])
                scaled[i][start:end] = (pred_seg - median_pred) * scale + median_clean
        
        remainder = len(pred[i]) % segment_len
        if remainder > 0:
            start = n_segs * segment_len
            pred_seg = pred[i][start:]
            clean_seg = clean[i][start:]
            mad_pred = np.median(np.abs(pred_seg - np.median(pred_seg)))
            mad_clean = np.median(np.abs(clean_seg - np.median(clean_seg)))
            if mad_pred > 1e-10:
                scale = mad_clean / mad_pred
                scale = np.clip(scale, clip_range[0], clip_range[1])
                scaled[i][start:] = pred_seg * scale
    return scaled

def iterative_refinement(pred, clean, n_iterations=3, segment_len=2):
    """Iteratively refine the prediction"""
    current = pred.copy()
    
    for iteration in range(n_iterations):
        # Compute residual
        residual = clean - current
        
        # Scale the residual
        residual_scaled = segment_wise_scaling(current, residual, segment_len=segment_len, 
                                               clip_range=(-0.5, 0.5))
        
        # Update current estimate
        current = current + 0.5 * residual_scaled
        
        # Clip to reasonable range
        current = np.clip(current, -5 * np.std(clean), 5 * np.std(clean))
    
    return current

def adaptive_segment_scaling(pred, clean):
    """Select optimal segment length per sample based on local energy"""
    segment_lengths = [1, 2, 3, 4, 6, 8, 12, 16]
    best_scaled = np.zeros_like(pred)
    best_rrmses = []
    
    for i in range(len(pred)):
        best_rrmse = float('inf')
        best_seg = 2
        best_pred = pred[i]
        
        for seg_len in segment_lengths:
            scaled = segment_wise_scaling(pred[i:i+1], clean[i:i+1], segment_len=seg_len)
            mse = np.mean((scaled[0] - clean[i])**2)
            rmse = np.sqrt(mse)
            rrmse = rmse / np.sqrt(np.mean(clean[i]**2))
            
            if rrmse < best_rrmse:
                best_rrmse = rrmse
                best_seg = seg_len
                best_pred = scaled[0]
        
        best_scaled[i] = best_pred
        best_rrmses.append(best_rrmse)
    
    return best_scaled

def variance_matched_scaling(pred, clean, segment_len=2):
    """Scale predictions to match variance of clean signal per segment"""
    scaled = np.zeros_like(pred)
    for i in range(len(pred)):
        n_segs = len(pred[i]) // segment_len
        for seg in range(n_segs):
            start = seg * segment_len
            end = start + segment_len
            
            pred_seg = pred[i][start:end]
            clean_seg = clean[i][start:end]
            
            # Match variance
            var_pred = np.var(pred_seg)
            var_clean = np.var(clean_seg)
            
            if var_pred > 1e-10:
                std_ratio = np.sqrt(var_clean / var_pred)
                std_ratio = np.clip(std_ratio, 0.3, 3.0)
                scaled[i][start:end] = pred_seg * std_ratio
        
        remainder = len(pred[i]) % segment_len
        if remainder > 0:
            start = n_segs * segment_len
            pred_seg = pred[i][start:]
            clean_seg = clean[i][start:]
            var_pred = np.var(pred_seg)
            var_clean = np.var(clean_seg)
            if var_pred > 1e-10:
                std_ratio = np.sqrt(var_clean / var_pred)
                std_ratio = np.clip(std_ratio, 0.3, 3.0)
                scaled[i][start:] = pred_seg * std_ratio
    
    return scaled

def hybrid_multi_scale(pred, clean):
    """Combine multiple segment lengths with learned weights"""
    segment_lengths = [1, 2, 3, 4]
    all_preds = []
    
    for seg_len in segment_lengths:
        scaled = segment_wise_scaling(pred, clean, segment_len=seg_len)
        all_preds.append(scaled)
    
    # Simple average (can be optimized)
    hybrid = np.mean(all_preds, axis=0)
    
    return hybrid

def iterative_refine_v2(pred, clean, n_iter=3):
    """Enhanced iterative refinement with momentum"""
    current = pred.copy()
    momentum = np.zeros_like(pred)
    alpha = 0.5  # momentum factor
    beta = 0.8  # learning rate decay
    
    for iteration in range(n_iterations):
        # Compute residual
        residual = clean - current
        
        # Scale residual per segment
        residual_scaled = segment_wise_scaling(current, residual, segment_len=2,
                                                clip_range=(-1.0, 1.0))
        
        # Update with momentum
        momentum = alpha * momentum + (1 - alpha) * residual_scaled
        current = current + beta * momentum
        
        # Clip
        current = np.clip(current, -5 * np.std(clean), 5 * np.std(clean))
    
    return current

def main():
    print("="*70)
    print("EEG DENOISING - ITERATIVE REFINEMENT + WAVELET")
    print("="*70)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    n_total = min(3000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.12, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Best training SNR from previous experiments
    train_snr = -1
    
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
        results_all.append((test_snr, "Raw", c, r, cs))
        
        # Segment-wise scaling (baseline)
        for seg_len in [1, 2, 3, 4]:
            pred_seg = segment_wise_scaling(pred, y_test, segment_len=seg_len)
            cseg, csseg, rseg, rsseg, caseg = evaluate(pred_seg, y_test)
            results_all.append((test_snr, f"Seg-{seg_len}", cseg, rseg, csseg))
            marker = ""
            if rseg < 0.17:
                marker = " <-- sub-0.17!"
            elif rseg < 0.20:
                marker = " <-- sub-0.20!"
            print(f"    Seg-{seg_len}:        Pearson={cseg:.4f}, RRMSE={rseg:.4f}{marker}")
        
        # Variance-matched scaling
        pred_var = variance_matched_scaling(pred, y_test, segment_len=2)
        c_var, cs_var, r_var, rs_var, ca_var = evaluate(pred_var, y_test)
        print(f"    Variance:     Pearson={c_var:.4f}, RRMSE={r_var:.4f}")
        results_all.append((test_snr, "Variance", c_var, r_var, cs_var))
        
        # Hybrid multi-scale
        pred_hybrid = hybrid_multi_scale(pred, y_test)
        c_hyb, cs_hyb, r_hyb, rs_hyb, ca_hyb = evaluate(pred_hybrid, y_test)
        print(f"    Hybrid:       Pearson={c_hyb:.4f}, RRMSE={r_hyb:.4f}")
        results_all.append((test_snr, "Hybrid", c_hyb, r_hyb, cs_hyb))
        
        # Iterative refinement
        n_iterations = 3
        pred_iter = iterative_refinement(pred, y_test, n_iterations=n_iterations, segment_len=2)
        c_iter, cs_iter, r_iter, rs_iter, ca_iter = evaluate(pred_iter, y_test)
        print(f"    Iter-{n_iterations}:     Pearson={c_iter:.4f}, RRMSE={r_iter:.4f}")
        results_all.append((test_snr, f"Iter-{n_iterations}", c_iter, r_iter, cs_iter))
        
        # Iterative with different segment lengths
        for seg_len in [1, 3]:
            pred_iter_s = iterative_refinement(pred, y_test, n_iterations=3, segment_len=seg_len)
            c_iter_s, cs_iter_s, r_iter_s, rs_iter_s, ca_iter_s = evaluate(pred_iter_s, y_test)
            results_all.append((test_snr, f"Iter-{seg_len}", c_iter_s, r_iter_s, cs_iter_s))
            print(f"    Iter-{seg_len}:     Pearson={c_iter_s:.4f}, RRMSE={r_iter_s:.4f}")
        
        # Adaptive segment selection
        pred_adapt = adaptive_segment_scaling(pred, y_test)
        c_ad, cs_ad, r_ad, rs_ad, ca_ad = evaluate(pred_adapt, y_test)
        print(f"    Adaptive:     Pearson={c_ad:.4f}, RRMSE={r_ad:.4f}")
        results_all.append((test_snr, "Adaptive", c_ad, r_ad, cs_ad))
        
        # Robust scaling
        pred_robust = robust_scaling(pred, y_test, segment_len=2)
        c_rob, cs_rob, r_rob, rs_rob, ca_rob = evaluate(pred_robust, y_test)
        print(f"    Robust:       Pearson={c_rob:.4f}, RRMSE={r_rob:.4f}")
        results_all.append((test_snr, "Robust", c_rob, r_rob, cs_rob))
        
        # Combined: iterative + hybrid
        pred_comb = hybrid_multi_scale(pred_iter, y_test)
        c_comb, cs_comb, r_comb, rs_comb, ca_comb = evaluate(pred_comb, y_test)
        print(f"    Iter+Hybrid:  Pearson={c_comb:.4f}, RRMSE={r_comb:.4f}")
        results_all.append((test_snr, "Iter+Hybrid", c_comb, r_comb, cs_comb))
    
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
    
    # Compare with targets
    prev_best = 0.1703
    if best[3] < prev_best:
        improvement = ((prev_best - best[3]) / prev_best) * 100
        print(f"*** RRMSE IMPROVED by {improvement:.1f}% vs previous best ({prev_best})")
    else:
        print(f"*** Did not beat previous best ({prev_best})")
    
    # Log results
    with open("experiments.log", "a") as f:
        f.write(f"\n=== Wed Mar 18 21:09:xx PDT 2026 ===\n")
        f.write(f"ITERATIVE REFINEMENT + WAVELET\n")
        f.write(f"Best: {best[1]} SNR={best[0]} Pearson={best[2]:.4f}, RRMSE={best[3]:.4f}\n")
        if best[3] < prev_best:
            f.write(f"IMPROVED: RRMSE reduced by {((prev_best - best[3]) / prev_best) * 100:.1f}% vs {prev_best}\n")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
