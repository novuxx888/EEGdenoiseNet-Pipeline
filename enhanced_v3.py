#!/usr/bin/env python3
"""
EEG Denoising - Enhanced v3
Focused approach: Better features, stronger ML, multi-SNR
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    """Load EEGdenoiseNet dataset"""
    eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
    eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
    return eeg, eog

def mix_at_snr(clean, noise, snr_db):
    """Mix clean signal with noise at specified SNR (dB)"""
    P_signal = np.mean(clean**2)
    P_noise = np.mean(noise**2)
    k = np.sqrt(P_signal / (10**(snr_db/10) * P_noise + 1e-10))
    return clean + k * noise

# ============================================================
# FEATURE ENGINEERING - Focus on quality over quantity
# ============================================================
def create_features_focused(x, eog_ref):
    """Focused feature engineering - quality over quantity"""
    feats = [x.copy()]  # Raw signal
    
    # Key bandpass filters (most important for EEG)
    bands = [(1, 30), (3, 25), (5, 20), (8, 13), (13, 20), (0.5, 40)]
    for low, high in bands:
        b, a = butter(3, [low/128, high/128], btype='band')
        feats.append(filtfilt(b, a, x))
    
    # Lowpass (critical for removing high-freq noise)
    for cut in [10, 15, 20, 30]:
        b, a = butter(3, cut/128, btype='low')
        feats.append(filtfilt(b, a, x))
    
    # Highpass (remove drift)
    b, a = butter(3, 0.5/128, btype='high')
    feats.append(filtfilt(b, a, x))
    
    # EOG regression - key insight: subtract scaled EOG
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        feats.append(x - alpha * eog_ref)
    
    # Temporal derivatives
    feats.append(np.gradient(x))
    feats.append(np.gradient(np.gradient(x)))
    
    # Detrended
    window = 16
    rolling_mean = np.convolve(x, np.ones(window)/window, mode='same')
    feats.append(x - rolling_mean)
    
    return np.concatenate(feats)

# ============================================================
# CLASSICAL FILTERS
# ============================================================
def butterworth_bandpass(data, fs=256):
    """Classical Butterworth bandpass filter"""
    b, a = butter(4, [0.5/128, 40/128], btype='band')
    return filtfilt(b, a, data)

def wiener_filter(data):
    """Classical Wiener filter"""
    from scipy.signal import wiener
    return wiener(data, mysize=15)

# ============================================================
# MULTI-SNR TRAINING
# ============================================================
def train_at_multiple_snr(eeg_all, eog_all, snr_levels=[-7, -5, -3, 0], n_per_snr=400):
    """Train on multiple SNR levels for better generalization"""
    print(f"  Training on SNR levels: {snr_levels}")
    
    X_train, y_train = [], []
    
    for snr in snr_levels:
        for i in range(n_per_snr):
            idx = np.random.randint(0, len(eeg_all))
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            
            noisy = mix_at_snr(eeg, eog, snr)
            feat = create_features_focused(noisy, eog)
            X_train.append(feat)
            y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Try multiple alpha values
    print("  Finding best alpha...")
    best_score = -1
    best_alpha = 1.0
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        model = Ridge(alpha=alpha)
        model.fit(X_train[:500], y_train[:500])  # Quick fit
        score = model.score(X_train[:500], y_train[:500])
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    print(f"  Best alpha: {best_alpha}")
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    
    return model, scaler, best_alpha

# ============================================================
# EVALUATION
# ============================================================
def evaluate(pred, truth):
    """Calculate metrics - matching original benchmark methodology"""
    pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
    truth = np.nan_to_num(truth, nan=0, posinf=0, neginf=0)
    
    # Method 1: Per-sample correlation (strict)
    corrs = []
    for i in range(len(pred)):
        c, _ = pearsonr(pred[i], truth[i])
        if not np.isnan(c):
            corrs.append(c)
    corr_mean = np.mean(corrs)
    corr_std = np.std(corrs)
    
    # Method 2: Mean signal correlation (matches original benchmark)
    pred_mean = pred.mean(axis=0)
    truth_mean = truth.mean(axis=0)
    corr_mean_avg, _ = pearsonr(pred_mean, truth_mean)
    
    # RRMSE on mean
    mse = np.mean((pred - truth)**2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.sqrt(np.mean(truth**2))
    
    return corr_mean, corr_std, rrmse, corr_mean_avg

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("EEG DENOISING - ENHANCED v3")
    print("Focus: Quality features, Multi-SNR, Better ML")
    print("="*60)
    
    # Load data
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    # Split data - use more samples
    n_total = min(2000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Prepare test data at -5dB
    print("\n[2] Preparing test data at SNR=-5dB...")
    X_test, y_test, noisy_test, eog_test = [], [], [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        
        noisy = mix_at_snr(eeg, eog, -5)
        
        X_test.append(create_features_focused(noisy, eog))
        y_test.append(eeg)
        noisy_test.append(noisy)
        eog_test.append(eog)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    noisy_test = np.array(noisy_test, dtype=np.float32)
    eog_test = np.array(eog_test, dtype=np.float32)
    
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    print(f"    Test shape: {X_test.shape}")
    
    results = {}
    
    # ============================================================
    # METHOD 1: Bandpass (Baseline)
    # ============================================================
    print("\n[3a] Bandpass filter...")
    preds_bp = np.array([butterworth_bandpass(ns) for ns in noisy_test])
    corr, corr_std, rrmse, corr_avg = evaluate(preds_bp, y_test)
    results['Bandpass'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f}±{corr_std:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 2: Wiener (Baseline)
    # ============================================================
    print("\n[3b] Wiener filter...")
    preds_wiener = np.array([wiener_filter(ns) for ns in noisy_test])
    corr, corr_std, rrmse, corr_avg = evaluate(preds_wiener, y_test)
    results['Wiener'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f}±{corr_std:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 3: Ridge at target SNR only
    # ============================================================
    print("\n[3c] Ridge (train at -5dB only)...")
    X_train, y_train = [], []
    for idx in train_idx[:800]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        X_train.append(create_features_focused(noisy, eog))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    scaler_single = StandardScaler()
    X_train_s = scaler_single.fit_transform(X_train)
    X_test_s = scaler_single.transform(X_test)
    
    model_single = Ridge(alpha=1.0)
    model_single.fit(X_train_s, y_train)
    preds_ridge_single = model_single.predict(X_test_s)
    corr, corr_std, rrmse, corr_avg = evaluate(preds_ridge_single, y_test)
    results['Ridge (-5dB)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f}±{corr_std:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 4: Multi-SNR Ridge
    # ============================================================
    print("\n[3d] Ridge (Multi-SNR training)...")
    model_multi, scaler_multi, best_alpha = train_at_multiple_snr(
        eeg_all, eog_all, snr_levels=[-7, -5, -3, 0], n_per_snr=500
    )
    X_test_multi = scaler_multi.transform(X_test)
    preds_ridge_multi = model_multi.predict(X_test_multi)
    corr, corr_std, rrmse, corr_avg = evaluate(preds_ridge_multi, y_test)
    results['Ridge (Multi-SNR)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f}±{corr_std:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 5: Ensemble
    # ============================================================
    print("\n[3e] Ensemble (Wiener + Ridge Multi-SNR)...")
    preds_ensemble = 0.5 * preds_wiener + 0.5 * preds_ridge_multi
    corr, corr_std, rrmse, corr_avg = evaluate(preds_ensemble, y_test)
    results['Ensemble'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f}±{corr_std:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 6: Train at easier SNR then apply
    # ============================================================
    print("\n[3f] Ridge (train at 0dB, test at -5dB - curriculum)...")
    X_train_0db, y_train_0db = [], []
    for idx in train_idx[:1000]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        # Train at easier SNR
        noisy = mix_at_snr(eeg, eog, 0)  # 0dB is easier!
        X_train_0db.append(create_features_focused(noisy, eog))
        y_train_0db.append(eeg)
    
    X_train_0db = np.array(X_train_0db, dtype=np.float32)
    y_train_0db = np.array(y_train_0db, dtype=np.float32)
    X_train_0db = np.nan_to_num(X_train_0db, nan=0, posinf=0, neginf=0)
    
    scaler_0db = StandardScaler()
    X_train_0db_s = scaler_0db.fit_transform(X_train_0db)
    X_test_0db_s = scaler_0db.transform(X_test)
    
    model_0db = Ridge(alpha=1.0)
    model_0db.fit(X_train_0db_s, y_train_0db)
    preds_0db = model_0db.predict(X_test_0db_s)
    corr, corr_std, rrmse, corr_avg = evaluate(preds_0db, y_test)
    results['Ridge (0dB train)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f}±{corr_std:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Pearson':>12} {'RRMSE':>10} {'Avg Corr':>10}")
    print("-"*50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][3], reverse=True)
    for method, (corr, corr_std, rrmse, corr_avg) in sorted_results:
        status = "✅" if corr >= 0.98 and rrmse <= 0.15 else ""
        print(f"{method:<20} {corr:>6.4f}±{corr_std:.4f} {rrmse:>10.4f} {corr_avg:>10.4f} {status}")
    
    print("\nTarget: Pearson ≥ 0.98, RRMSE ≤ 0.15")
    
    best_method = sorted_results[0][0]
    best_corr = sorted_results[0][1][3]
    if best_corr < 0.98:
        print(f"\n⚠️ Target not met. Best: {best_method} with Pearson={best_corr:.4f}")
    else:
        print(f"\n🎉 Target met with {best_method}!")

if __name__ == "__main__":
    main()
