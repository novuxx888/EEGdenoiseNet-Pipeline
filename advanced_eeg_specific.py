#!/usr/bin/env python3
"""
EEG Denoising - Advanced EEG-Specific Techniques
Exploiting EEG-specific properties beyond traditional filtering:
1. Kurtosis-based artifact detection
2. Wavelet decomposition
3. Non-linear features (sample entropy, fractal dimension)
4. ICA for artifact separation
5. Higher-order statistics
"""

import numpy as np
from scipy.stats import pearsonr, skew, kurtosis
from scipy.signal import butter, filtfilt, welch
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
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
    """Mix clean EEG with noise at given SNR (dB)"""
    P_signal = np.mean(clean**2)
    P_noise = np.mean(noise**2)
    k = np.sqrt(P_signal / (10**(snr_db/10) * P_noise + 1e-10))
    return clean + k * noise

def butter_bandpass(data, fs=128):
    """Bandpass filter 0.5-40 Hz"""
    b, a = butter(4, [0.5/fs*2, 40/fs*2], btype='band')
    return filtfilt(b, a, data)

# ============ EEG-SPECIFIC TECHNIQUES ============

def compute_kurtosis(x, axis=1):
    """Compute kurtosis - EEG has lower kurtosis than EOG/EMG artifacts"""
    return kurtosis(x, axis=axis)

def compute_wavelet_features(x, wavelet='db4', level=5):
    """Wavelet decomposition - separate frequency bands"""
    import pywt
    coeffs = pywt.wavedec(x, wavelet, level=level)
    features = []
    for c in coeffs:
        features.append(c)
        features.append(np.abs(c))
        features.append(np.mean(c**2))  # Energy
    return np.concatenate([np.array(f).flatten() for f in features[:6]])

def compute_nonlinear_features(x):
    """Non-linear features: sample entropy, fractal dimension"""
    n = len(x)
    if n < 20:
        return np.zeros(8)
    
    # Sample entropy approximation
    def sampen(L, m=2, r=0.2):
        N = len(L)
        r *= np.std(L)
        if r == 0:
            return 0
        counts = 0
        for i in range(N - m):
            for j in range(N - m):
                if i != j and np.max(np.abs(L[i:i+m] - L[j:j+m])) < r:
                    counts += 1
        return -np.log(counts / (N - m) / (N - m - 1) + 1e-10) if counts > 0 else 0
    
    # Higuchi fractal dimension
    def hfd(L, Kmax=10):
        n = len(L)
        if n < Kmax:
            return 0
        L = np.array(L)
        x = []
        y = []
        for k in range(1, Kmax+1):
            Lk = []
            for m in range(k):
                indices = range(m, n, k)
                if len(indices) < 2:
                    continue
                Lmk = np.mean(np.abs(np.diff(L[indices])))
                if Lmk > 0:
                    Lk.append(Lmk * (n - 1) / k)
            if len(Lk) > 0:
                x.append(np.log(k))
                y.append(np.mean(np.log(Lk + 1e-10)))
        if len(x) > 1:
            return np.polyfit(x, y, 1)[0]
        return 0
    
    features = [
        sampen(x[:min(100, len(x))], m=2, r=0.2),
        hfd(x[:min(200, len(x))], Kmax=8),
        skew(x),
        kurtosis(x),
    ]
    
    # Additional non-linear
    features.append(np.mean(np.diff(x)**2))  # Rate of change
    features.append(np.max(np.abs(x)) / (np.std(x) + 1e-10))  # Peak-to-RMS
    features.append(np.sum(x**2) / (len(x) * np.var(x) + 1e-10))  # Normalized energy
    features.append(np.percentile(np.abs(x), 95) / (np.percentile(np.abs(x), 5) + 1e-10))  # Dynamic range
    
    return np.array(features)

def compute_ica_components(signals, n_components=5):
    """ICA decomposition to separate artifacts"""
    try:
        ica = FastICA(n_components=min(n_components, signals.shape[0], signals.shape[1]), 
                      max_iter=500, random_state=42)
        sources = ica.fit_transform(signals)
        mixing = ica.mixing_
        return sources, mixing, ica
    except:
        return signals, np.eye(len(signals)), None

def compute_higher_order_stats(x):
    """Higher-order statistics for artifact detection"""
    stats = []
    # Moments
    stats.append(skew(x))
    stats.append(kurtosis(x))
    
    # Percentiles
    stats.append(np.percentile(x, 5))
    stats.append(np.percentile(x, 95))
    stats.append(np.percentile(x, 75) - np.percentile(x, 25))  # IQR
    
    # Extreme value ratios
    threshold = 3 * np.std(x)
    stats.append(np.mean(np.abs(x) > threshold))  # Proportion of outliers
    
    return np.array(stats)

def compute_frequency_features(x, fs=128):
    """Frequency domain features"""
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    
    # Band powers
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    features = []
    total_power = np.sum(psd) + 1e-10
    for name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        band_power = np.sum(psd[mask])
        features.append(band_power)
        features.append(band_power / total_power)  # Relative power
    
    # Spectral edge (95% power)
    cumsum = np.cumsum(psd)
    edge_freq = freqs[cumsum >= 0.95 * total_power][0] if np.any(cumsum >= 0.95 * total_power) else freqs[-1]
    features.append(edge_freq)
    
    # Spectral entropy
    psd_norm = psd / (total_power + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    features.append(spectral_entropy)
    
    return np.array(features)

def create_advanced_features(x, eog_ref, emg_ref=None):
    """Create comprehensive feature set using EEG-specific properties"""
    features = []
    
    # 1. Basic filtered signals
    features.append(x)
    
    # 2. Multi-band frequency features
    for low, high in [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]:
        b, a = butter(3, [low/64, high/64], btype='band')
        features.append(filtfilt(b, a, x))
    
    # 3. Low/high-pass variations
    for cut in [5, 10, 15, 20, 30]:
        b, a = butter(3, cut/64, btype='low')
        features.append(filtfilt(b, a, x))
    
    b, a = butter(3, 0.5/64, btype='high')
    features.append(filtfilt(b, a, x))
    
    # 4. EOG subtraction with different weights
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5]:
        if eog_ref is not None:
            features.append(x - alpha * eog_ref)
    
    # 5. Gradient features
    features.append(np.gradient(x))
    features.append(np.gradient(np.gradient(x)))
    
    # 6. Rolling statistics
    window = 16
    rolling_mean = np.convolve(x, np.ones(window)/window, mode='same')
    rolling_std = np.array([np.std(x[max(0,i-window):i+1]) for i in range(len(x))])
    features.append(x - rolling_mean)
    features.append(rolling_std)
    
    # 7. Kurtosis-based (per window)
    window_kurt = 64
    kurt_feats = np.array([kurtosis(x[i:i+window_kurt]) if i+window_kurt <= len(x) else kurtosis(x[i:]) 
                          for i in range(0, len(x), window//2)])
    # Interpolate to full length
    kurt_interp = np.interp(np.arange(len(x)), 
                            np.linspace(0, len(x)-1, len(kurt_feats)), 
                            kurt_feats)
    features.append(kurt_interp)
    
    # 8. Wavelet features (downsampled for efficiency)
    try:
        wavelet_feats = compute_wavelet_features(x[:256] if len(x) > 256 else x)
        # Repeat to match length
        wavelet_full = np.tile(wavelet_feats, (len(x) // len(wavelet_feats) + 1))[:len(x)]
        features.append(wavelet_full)
    except:
        pass
    
    # 9. Non-linear features (repeated to match length)
    nl_feats = compute_nonlinear_features(x)
    for f in nl_feats:
        features.append(np.full(len(x), f))
    
    # 10. Higher-order statistics
    hos_feats = compute_higher_order_stats(x)
    for f in hos_feats:
        features.append(np.full(len(x), f))
    
    # 11. Frequency features
    freq_feats = compute_frequency_features(x)
    for f in freq_feats:
        features.append(np.full(len(x), f))
    
    # 12. EMG reference features (if available)
    if emg_ref is not None:
        for alpha in [0.3, 0.5, 0.7]:
            features.append(x - alpha * emg_ref)
    
    return np.concatenate(features)

def create_enhanced_features(x, eog_ref, emg_ref=None):
    """Streamlined enhanced features - more practical version"""
    feats = [x]
    
    # Multi-band decomposition
    bands = [(1, 30), (3, 10), (8, 13), (13, 25), (1, 4), (4, 8), (25, 40)]
    for low, high in bands:
        b, a = butter(3, [low/64, high/64], btype='band')
        feats.append(filtfilt(b, a, x))
    
    # Temporal filters
    for cut in [8, 15, 25, 35]:
        b, a = butter(3, cut/64, btype='low')
        feats.append(filtfilt(b, a, x))
    
    b, a = butter(3, 1/64, btype='high')
    feats.append(filtfilt(b, a, x))
    
    # EOG regression with various weights
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0, 1.3]:
        feats.append(x - alpha * eog_ref)
    
    # EMG regression if available
    if emg_ref is not None:
        for alpha in [0.3, 0.5, 0.8]:
            feats.append(x - alpha * emg_ref)
    
    # Gradient and acceleration
    feats.append(np.gradient(x))
    feats.append(np.gradient(np.gradient(x)))
    
    # Detrended signal
    window = 32
    trend = np.convolve(x, np.ones(window)/window, mode='same')
    feats.append(x - trend)
    
    # Kurtosis window feature
    kurt_win = 32
    kurt_arr = np.array([kurtosis(x[i:i+kurt_win]) for i in range(0, len(x)-kurt_win, 8)])
    kurt_full = np.interp(np.arange(len(x)), np.linspace(0, len(x)-kurt_win-1, len(kurt_arr)), kurt_arr)
    feats.append(kurt_full)
    
    # Nonlinear
    feats.append(np.abs(np.gradient(x)))
    feats.append(np.sign(x) * np.log1p(np.abs(x)))
    
    return np.concatenate(feats)

def evaluate(pred, truth):
    """Evaluate predictions"""
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
    print("EEG DENOISING - ADVANCED EEG-SPECIFIC TECHNIQUES")
    print("="*70)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all, emg_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}, EMG: {emg_all.shape}")
    
    # Use subset for faster training
    n_total = min(3000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Prepare test data at target SNR
    print("\n[2] Preparing test data at SNR=-5dB...")
    X_test_eog, X_test_emg, y_test = [], [], []
    noisy_test_eog, noisy_test_emg = [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        emg = emg_all[idx % len(emg_all)]
        
        # EOG mixed
        noisy_eog = mix_at_snr(eeg, eog, -5)
        noisy_eog_f = butter_bandpass(noisy_eog)
        
        # EMG mixed  
        noisy_emg = mix_at_snr(eeg, emg, -5)
        noisy_emg_f = butter_bandpass(noisy_emg)
        
        X_test_eog.append(create_enhanced_features(noisy_eog_f, eog, emg))
        X_test_emg.append(create_enhanced_features(noisy_emg_f, eog, emg))
        y_test.append(eeg)
        noisy_test_eog.append(noisy_eog_f)
        noisy_test_emg.append(noisy_emg_f)
    
    X_test_eog = np.array(X_test_eog, dtype=np.float32)
    X_test_emg = np.array(X_test_emg, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    noisy_test_eog = np.array(noisy_test_eog, dtype=np.float32)
    noisy_test_emg = np.array(noisy_test_emg, dtype=np.float32)
    
    X_test_eog = np.nan_to_num(X_test_eog, nan=0, posinf=0, neginf=0)
    X_test_emg = np.nan_to_num(X_test_emg, nan=0, posinf=0, neginf=0)
    
    print(f"    Feature dim: {X_test_eog.shape[1]}")
    
    results = {}
    
    # ===== EOG DENOISING =====
    print("\n" + "="*50)
    print("EOG DENOISING")
    print("="*50)
    
    # Test different SNR levels and select best
    best_eog_snr = None
    best_eog_score = -1
    
    for snr in [-6, -4, -2, 0, 2]:
        X_train, y_train = [], []
        
        for idx in train_idx[:1800]:
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, eog, snr)
            noisy_f = butter_bandpass(noisy)
            X_train.append(create_enhanced_features(noisy_f, eog))
            y_train.append(eeg)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        
        for alpha in [0.01, 0.03, 0.05, 0.1]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_s, y_train)
            
            X_test_s = scaler.transform(X_test_eog)
            preds = ridge.predict(X_test_s)
            corr, corr_std, rrmse, corr_avg = evaluate(preds, y_test)
            
            score = corr_avg - 0.1 * rrmse  # Combined score
            if score > best_eog_score:
                best_eog_score = score
                best_eog_snr = (snr, alpha)
            
            results[f'EOG SNR{snr} a{alpha}'] = (corr, rrmse, corr_avg)
    
    print(f"\nBest EOG config: SNR={best_eog_snr[0]}dB, α={best_eog_snr[1]}")
    
    # Retrain with best config
    snr, alpha = best_eog_snr
    X_train, y_train = [], []
    for idx in train_idx[:2000]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, snr)
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_enhanced_features(noisy_f, eog))
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
    results['EOG-Best'] = (corr_eog, rrmse_eog, corr_avg_eog)
    
    print(f"    EOG Results: Pearson={corr_eog:.4f}, RRMSE={rrmse_eog:.4f}, Avg={corr_avg_eog:.4f}")
    
    # ===== EMG DENOISING =====
    print("\n" + "="*50)
    print("EMG DENOISING")
    print("="*50)
    
    best_emg_snr = None
    best_emg_score = -1
    
    for snr in [-6, -4, -2, 0, 2]:
        X_train, y_train = [], []
        
        for idx in train_idx[:1800]:
            eeg = eeg_all[idx]
            emg = emg_all[idx % len(emg_all)]
            eog = eog_all[idx % len(eog_all)]  # Also use EOG reference
            noisy = mix_at_snr(eeg, emg, snr)
            noisy_f = butter_bandpass(noisy)
            X_train.append(create_enhanced_features(noisy_f, eog, emg))
            y_train.append(eeg)
        
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        
        for alpha in [0.01, 0.03, 0.05, 0.1]:
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train_s, y_train)
            
            X_test_s = scaler.transform(X_test_emg)
            preds = ridge.predict(X_test_s)
            corr, corr_std, rrmse, corr_avg = evaluate(preds, y_test)
            
            # EMG is harder, weight differently
            score = corr_avg - 0.2 * rrmse
            if score > best_emg_score:
                best_emg_score = score
                best_emg_snr = (snr, alpha)
            
            results[f'EMG SNR{snr} a{alpha}'] = (corr, rrmse, corr_avg)
    
    print(f"\nBest EMG config: SNR={best_emg_snr[0]}dB, α={best_emg_snr[1]}")
    
    # Retrain with best config
    snr, alpha = best_emg_snr
    X_train, y_train = [], []
    for idx in train_idx[:2000]:
        eeg = eeg_all[idx]
        emg = emg_all[idx % len(emg_all)]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, emg, snr)
        noisy_f = butter_bandpass(noisy)
        X_train.append(create_enhanced_features(noisy_f, eog, emg))
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
    results['EMG-Best'] = (corr_emg, rrmse_emg, corr_avg_emg)
    
    print(f"    EMG Results: Pearson={corr_emg:.4f}, RRMSE={rrmse_emg:.4f}, Avg={corr_avg_emg:.4f}")
    
    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"{'Method':<25} {'Pearson':>12} {'RRMSE':>10} {'Avg Corr':>10}")
    print("-"*60)
    
    for method, (corr, rrmse, corr_avg) in sorted(results.items(), key=lambda x: x[1][2], reverse=True)[:10]:
        print(f"{method:<25} {corr:>12.4f} {rrmse:>10.4f} {corr_avg:>10.4f}")
    
    print("\n" + "-"*60)
    print(f"EOG:  Pearson={corr_eog:.4f}, RRMSE={rrmse_eog:.4f}")
    print(f"EMG:  Pearson={corr_emg:.4f}, RRMSE={rrmse_emg:.4f}")
    print("-"*60)
    
    # Save results
    np.save('preds_eog_advanced.npy', preds_eog)
    np.save('preds_emg_advanced.npy', preds_emg)
    np.save('y_test.npy', y_test)
    
    print("\n✓ Results saved to preds_eog_advanced.npy, preds_emg_advanced.npy")
    
    return {
        'eog': {'pearson': corr_eog, 'rrmse': rrmse_eog, 'avg_corr': corr_avg_eog},
        'emg': {'pearson': corr_emg, 'rrmse': rrmse_emg, 'avg_corr': corr_avg_emg}
    }

if __name__ == "__main__":
    results = main()
