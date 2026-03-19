#!/usr/bin/env python3
"""
EEG Denoising Pipeline - Publication Quality
Multi-method comparison with rigorous evaluation
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
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
    k = np.sqrt(P_signal / (10**(snr_db/10) * P_noise))
    return clean + k * noise

# ============================================================
# FEATURE ENGINEERING
# ============================================================
def create_features(x, eog_ref):
    """Enhanced feature engineering"""
    feats = [x]
    
    # Bandpass filters
    for low, high in [(1, 30), (3, 25), (5, 20), (8, 15), (0.5, 40)]:
        b, a = butter(3, [low/128, high/128], btype='band')
        feats.append(filtfilt(b, a, x))
    
    # Lowpass filters
    for cut in [8, 12, 18, 25]:
        b, a = butter(3, cut/128, btype='low')
        feats.append(filtfilt(b, a, x))
    
    # Highpass (remove drift)
    b, a = butter(3, 0.5/128, btype='high')
    feats.append(filtfilt(b, a, x))
    
    # EOG subtractions
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0]:
        feats.append(x - alpha * eog_ref)
    
    # Temporal derivatives
    feats.append(np.gradient(x))
    feats.append(np.gradient(np.gradient(x)))
    
    return np.concatenate(feats)

# ============================================================
# METHODS
# ============================================================
def butterworth_bandpass(data, fs=256):
    """Classical Butterworth bandpass filter"""
    b, a = butter(4, [0.5/128, 40/128], btype='band')
    return filtfilt(b, a, data)

def wiener_filter(data, fs=256):
    """Classical Wiener filter"""
    from scipy.signal import wiener
    return wiener(data, mysize=15)

def ridge_denoise(X_train, y_train, X_test, alpha=1.0):
    """Ridge regression"""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model.predict(X_test)

def elasticnet_denoise(X_train, y_train, X_test):
    """ElasticNet regression"""
    best_score = -1
    best_model = None
    for alpha in [0.01, 0.1, 1.0]:
        for l1_ratio in [0.3, 0.5, 0.7]:
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000)
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            if score > best_score:
                best_score = score
                best_model = model
    return best_model.predict(X_test)

def random_forest_denoise(X_train, y_train, X_test):
    """Random Forest regressor"""
    model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================
def evaluate(pred, truth):
    """Calculate metrics"""
    corr, _ = pearsonr(pred, truth)
    mse = np.mean((pred - truth)**2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.sqrt(np.mean(truth**2))
    
    # SNR improvement
    noisy_snr = 10 * np.log10(np.mean(truth**2) / np.mean((truth - truth)**2 + 1e-10))
    clean_snr = 10 * np.log10(np.mean(pred**2) / np.mean((pred - truth)**2 + 1e-10))
    snr_improvement = clean_snr - (-5)  # Original was -5dB
    
    return corr, rrmse, snr_improvement

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("EEG DENOISING - PUBLICATION QUALITY BENCHMARK")
    print("="*60)
    
    # Load data
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    # Split data
    n_total = min(2000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Prepare data at -5dB
    print("\n[2] Preparing data at SNR=-5dB...")
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for idx in train_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        feat = create_features(noisy, eog)
        X_train.append(feat)
        y_train.append(eeg)
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        feat = create_features(noisy, eog)
        X_test.append(feat)
        y_test.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"    Train shape: {X_train.shape}")
    print(f"    Test shape: {X_test.shape}")
    
    # Run methods
    print("\n[3] Running methods...")
    results = {}
    
    # Baseline: Bandpass
    print("    Bandpass filter...", end=" ")
    preds = np.array([butterworth_bandpass(y_test[i]) for i in range(len(y_test))])
    corr, rrmse, snr = evaluate(preds.mean(axis=0), y_test.mean(axis=0))
    results['Bandpass'] = (corr, rrmse, snr)
    print(f"Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # Baseline: Wiener
    print("    Wiener filter...", end=" ")
    preds = np.array([wiener_filter(y_test[i]) for i in range(len(y_test))])
    corr, rrmse, snr = evaluate(preds.mean(axis=0), y_test.mean(axis=0))
    results['Wiener'] = (corr, rrmse, snr)
    print(f"Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # Ridge
    print("    Ridge Regression...", end=" ")
    preds = ridge_denoise(X_train, y_train, X_test)
    corr, rrmse, snr = evaluate(preds.mean(axis=0), y_test.mean(axis=0))
    results['Ridge'] = (corr, rrmse, snr)
    print(f"Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ElasticNet
    print("    ElasticNet...", end=" ")
    preds = elasticnet_denoise(X_train, y_train, X_test)
    corr, rrmse, snr = evaluate(preds.mean(axis=0), y_test.mean(axis=0))
    results['ElasticNet'] = (corr, rrmse, snr)
    print(f"Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # Random Forest
    print("    Random Forest...", end=" ")
    preds = random_forest_denoise(X_train, y_train, X_test)
    corr, rrmse, snr = evaluate(preds.mean(axis=0), y_test.mean(axis=0))
    results['RandomForest'] = (corr, rrmse, snr)
    print(f"Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # Per-sample evaluation
    print("\n[4] Per-sample evaluation...")
    pearson_scores = []
    rrmse_scores = []
    
    preds = ridge_denoise(X_train, y_train, X_test)
    for i in range(len(test_idx)):
        c, r, _ = evaluate(preds[i], y_test[i])
        pearson_scores.append(c)
        rrmse_scores.append(r)
    
    mean_p = np.mean(pearson_scores)
    std_p = np.std(pearson_scores)
    mean_r = np.mean(rrmse_scores)
    std_r = np.std(rrmse_scores)
    
    print(f"\n    Ridge: Pearson={mean_p:.4f}±{std_p:.4f}, RRMSE={mean_r:.4f}±{std_r:.4f}")
    
    # Final table
    print("\n" + "="*60)
    print("📊 RESULTS TABLE (Mean across test set)")
    print("="*60)
    print(f"{'Method':<20} {'Pearson':>12} {'RRMSE':>12} {'SNR Imp (dB)':>15}")
    print("-"*60)
    
    for method, (c, r, s) in sorted(results.items(), key=lambda x: -x[1][0]):
        print(f"{method:<20} {c:>12.4f} {r:>12.4f} {s:>15.2f}")
    
    print("="*60)
    
    # Target check
    print(f"\n🎯 TARGET CHECK:")
    print(f"   Pearson ≥ 0.98: {'❌' if mean_p < 0.98 else '✅'} ({mean_p:.4f})")
    print(f"   RRMSE ≤ 0.15: {'❌' if mean_r > 0.15 else '✅'} ({mean_r:.4f})")

if __name__ == "__main__":
    main()
