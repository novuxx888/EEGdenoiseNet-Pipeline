#!/usr/bin/env python3
"""
EEG Denoising - HistGradientBoosting
Fast ML that can handle large feature sets
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# DATA LOADING
# ============================================================
def load_data():
    eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
    eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
    return eeg, eog

def mix_at_snr(clean, noise, snr_db):
    P_signal = np.mean(clean**2)
    P_noise = np.mean(noise**2)
    k = np.sqrt(P_signal / (10**(snr_db/10) * P_noise + 1e-10))
    return clean + k * noise

# ============================================================
# FEATURES
# ============================================================
def create_features(x, eog_ref):
    """Focused features"""
    feats = [x]
    
    # Bandpass filters
    for low, high in [(1, 30), (3, 25), (5, 20), (8, 13), (13, 20), (0.5, 40)]:
        b, a = butter(3, [low/128, high/128], btype='band')
        feats.append(filtfilt(b, a, x))
    
    # Lowpass
    for cut in [10, 15, 20, 30]:
        b, a = butter(3, cut/128, btype='low')
        feats.append(filtfilt(b, a, x))
    
    # Highpass
    b, a = butter(3, 0.5/128, btype='high')
    feats.append(filtfilt(b, a, x))
    
    # EOG regression
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.2, 1.5]:
        feats.append(x - alpha * eog_ref)
    
    # Temporal
    feats.append(np.gradient(x))
    feats.append(np.gradient(np.gradient(x)))
    
    # Detrended
    window = 16
    rolling_mean = np.convolve(x, np.ones(window)/window, mode='same')
    feats.append(x - rolling_mean)
    
    return np.concatenate(feats)

def butter_bandpass(data):
    b, a = butter(4, [0.5/128, 40/128], btype='band')
    return filtfilt(b, a, data)

# ============================================================
# EVALUATION
# ============================================================
def evaluate(pred, truth):
    pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
    truth = np.nan_to_num(truth, nan=0, posinf=0, neginf=0)
    
    # Per-sample
    corrs = []
    for i in range(len(pred)):
        c, _ = pearsonr(pred[i], truth[i])
        if not np.isnan(c):
            corrs.append(c)
    corr_mean = np.mean(corrs)
    corr_std = np.std(corrs)
    
    # Mean signal
    corr_avg, _ = pearsonr(pred.mean(axis=0), truth.mean(axis=0))
    
    mse = np.mean((pred - truth)**2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.sqrt(np.mean(truth**2))
    
    return corr_mean, corr_std, rrmse, corr_avg

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("EEG DENOISING - HISTGRADIENT BOOSTING")
    print("="*60)
    
    # Load
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    # Split
    n_total = min(2000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Test data at -5dB
    print("\n[2] Preparing test data at SNR=-5dB...")
    X_test, y_test, noisy_test = [], [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_filtered = butter_bandpass(noisy)
        X_test.append(create_features(noisy_filtered, eog))
        y_test.append(eeg)
        noisy_test.append(noisy_filtered)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    noisy_test = np.array(noisy_test, dtype=np.float32)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    print(f"    Test shape: {X_test.shape}")
    
    results = {}
    
    # ============================================================
    # BASELINE
    # ============================================================
    print("\n[3a] Bandpass baseline...")
    corr, corr_std, rrmse, corr_avg = evaluate(noisy_test, y_test)
    results['Bandpass'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f})")
    
    # ============================================================
    # RIDGE (best so far)
    # ============================================================
    print("\n[3b] Ridge (train 0dB, curriculum)...")
    X_train, y_train = [], []
    for idx in train_idx[:1000]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, 0)  # Easier SNR
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_features(noisy_f, eog))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    ridge = Ridge(alpha=0.1)
    ridge.fit(X_train_s, y_train)
    preds_ridge = ridge.predict(X_test_s)
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_ridge, y_test)
    results['Ridge (0dB)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # HISTGRADIENT BOOSTING
    # ============================================================
    print("\n[3c] HistGradientBoostingRegressor (curriculum)...")
    print("    Training...")
    
    # Use subset for faster training
    X_train_hgb, y_train_hgb = [], []
    for idx in train_idx[:800]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        # Train at multiple SNRs for curriculum
        for snr in [0, 3]:
            noisy = mix_at_snr(eeg, eog, snr)
            noisy_f = butter_bandpass(noisy)
            X_train_hgb.append(create_features(noisy_f, eog))
            y_train_hgb.append(eeg)
    
    X_train_hgb = np.array(X_train_hgb, dtype=np.float32)
    y_train_hgb = np.array(y_train_hgb, dtype=np.float32)
    X_train_hgb = np.nan_to_num(X_train_hgb, nan=0, posinf=0, neginf=0)
    
    print(f"    Training on {len(X_train_hgb)} samples...")
    
    hgb = HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=8,
        learning_rate=0.1,
        random_state=42
    )
    hgb.fit(X_train_hgb, y_train_hgb)
    
    preds_hgb = hgb.predict(X_test)
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_hgb, y_test)
    results['HGB (MultiSNR)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # ENSEMBLE: Ridge + HGB
    # ============================================================
    print("\n[3d] Ensemble (Ridge + HGB)...")
    preds_ens = 0.5 * preds_ridge + 0.5 * preds_hgb
    corr, corr_std, rrmse, corr_avg = evaluate(preds_ens, y_test)
    results['Ensemble'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # HGB: More data, harder curriculum
    # ============================================================
    print("\n[3e] HGB (more data + harder curriculum)...")
    X_train_hgb2, y_train_hgb2 = [], []
    for idx in train_idx[:1200]:  # More data
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        # Curriculum: easier to harder
        for snr in [-3, 0, 3, 5]:
            noisy = mix_at_snr(eeg, eog, snr)
            noisy_f = butter_bandpass(noisy)
            X_train_hgb2.append(create_features(noisy_f, eog))
            y_train_hgb2.append(eeg)
    
    X_train_hgb2 = np.array(X_train_hgb2, dtype=np.float32)
    y_train_hgb2 = np.array(y_train_hgb2, dtype=np.float32)
    X_train_hgb2 = np.nan_to_num(X_train_hgb2, nan=0, posinf=0, neginf=0)
    
    print(f"    Training on {len(X_train_hgb2)} samples...")
    
    hgb2 = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=10,
        learning_rate=0.08,
        l2_regularization=0.1,
        random_state=42
    )
    hgb2.fit(X_train_hgb2, y_train_hgb2)
    
    preds_hgb2 = hgb2.predict(X_test)
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_hgb2, y_test)
    results['HGB (Large)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
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
    
    best = sorted_results[0]
    print(f"\nBest: {best[0]} with Avg Pearson={best[1][3]:.4f}")

if __name__ == "__main__":
    main()
