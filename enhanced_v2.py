#!/usr/bin/env python3
"""
EEG Denoising - Enhanced v2
Novel approaches: Wavelet, Multi-SNR Training, Adaptive Filters
"""

import numpy as np
from scipy import signal
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt, lfilter
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
    """Load EEGdenoiseNet dataset"""
    eeg = np.load("data/EEG_all_epochs.npy", allow_pickle=True)
    eog = np.load("data/EOG_all_epochs.npy", allow_pickle=True)
    emg = np.load("data/EMG_all_epochs.npy", allow_pickle=True)
    return eeg, eog, emg

def mix_at_snr(clean, noise, snr_db):
    """Mix clean signal with noise at specified SNR (dB)"""
    P_signal = np.mean(clean**2)
    P_noise = np.mean(noise**2)
    k = np.sqrt(P_signal / (10**(snr_db/10) * P_noise + 1e-10))
    return clean + k * noise

# ============================================================
# WAVELET DENOISING (Novel Approach 1)
# ============================================================
def wavelet_denoise(data, wavelet='db4', level=4, threshold_mode='soft'):
    """Wavelet-based denoising using PyWavelets"""
    try:
        import pywt
        # Decompose
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # Estimate noise sigma from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        
        # Universal threshold (VisuShrink)
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        
        # Threshold detail coefficients (keep approximation intact)
        thresholded = [coeffs[0]]  # Keep approximation
        for c in coeffs[1:]:
            thresholded.append(pywt.threshold(c, threshold, mode=threshold_mode))
        
        # Reconstruct
        return pywt.waverec(thresholded, wavelet)[:len(data)]
    except:
        # Fallback to simple smoothing if pywt not available
        return butterworth_bandpass(data)

def wavelet_denoise_batch(data, wavelet='db4', level=4):
    """Apply wavelet denoising to batch"""
    return np.array([wavelet_denoise(d, wavelet, level) for d in data])

# ============================================================
# ADAPTIVE FILTERING (Novel Approach 2)
# ============================================================
def lms_filter(noisy, reference, mu=0.1, n_taps=32):
    """LMS adaptive filter"""
    n = len(noisy)
    w = np.zeros(n_taps)
    output = np.zeros(n)
    error = np.zeros(n)
    
    # Normalize reference
    reference = (reference - np.mean(reference)) / (np.std(reference) + 1e-10)
    
    for i in range(n_taps, n):
        x = reference[i-n_taps:i][::-1]
        y = np.dot(w, x)
        output[i] = y
        error[i] = noisy[i] - y
        w = w + mu * error[i] * x
    
    result = noisy - output
    return np.nan_to_num(result, nan=0, posinf=0, neginf=0)

def rls_filter(noisy, reference, lam=0.99, n_taps=32):
    """RLS adaptive filter (simplified)"""
    n = len(noisy)
    w = np.zeros(n_taps)
    P = np.eye(n_taps) * 1.0
    output = np.zeros(n)
    
    for i in range(n_taps, n):
        x = reference[i-n_taps:i][::-1]
        y = np.dot(w, x)
        output[i] = y
        
        # RLS update
        k = np.dot(P, x) / (lam + np.dot(x, np.dot(P, x)))
        e = noisy[i] - y
        w = w + k * e
        P = (P - np.outer(k, x) @ P) / lam
    
    return noisy - output

def adaptive_filter_batch(noisy, reference, method='lms', mu=0.05):
    """Apply adaptive filter to batch"""
    results = []
    for i in range(len(noisy)):
        if method == 'lms':
            result = lms_filter(noisy[i], reference[i], mu=mu)
        else:
            result = rls_filter(noisy[i], reference[i], lam=0.99)
        results.append(result)
    return np.array(results)

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

def temporal_median(data, window=5):
    """Temporal median filter for spike removal"""
    from scipy.ndimage import median_filter
    return median_filter(data, size=window)

# ============================================================
# FEATURE ENGINEERING (Enhanced)
# ============================================================
def create_features_enhanced(x, eog_ref, emg_ref=None):
    """Enhanced feature engineering with more filters"""
    feats = [x]
    
    # Bandpass filters at different frequency bands
    bands = [(0.5, 40), (1, 30), (3, 25), (5, 20), (8, 13), (13, 20)]
    for low, high in bands:
        b, a = butter(3, [low/128, high/128], btype='band')
        feats.append(filtfilt(b, a, x))
    
    # Lowpass filters
    for cut in [8, 15, 20, 30]:
        b, a = butter(3, cut/128, btype='low')
        feats.append(filtfilt(b, a, x))
    
    # Highpass (remove drift)
    b, a = butter(3, 0.3/128, btype='high')
    feats.append(filtfilt(b, a, x))
    
    # EOG subtraction with multiple alphas
    for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        feats.append(x - alpha * eog_ref)
    
    # EMG subtraction if available
    if emg_ref is not None:
        for alpha in [0.3, 0.5, 0.7, 1.0]:
            feats.append(x - alpha * emg_ref)
    
    # Temporal features
    feats.append(np.gradient(x))
    feats.append(np.gradient(np.gradient(x)))
    
    # Rolling statistics
    window = 16
    rolling_mean = np.convolve(x, np.ones(window)/window, mode='same')
    feats.append(x - rolling_mean)  # Detrended
    
    return np.concatenate(feats)

# ============================================================
# MULTI-SNR TRAINING (Novel Approach 3)
# ============================================================
def train_multisnr_model(eeg_all, eog_all, emg_all, snr_levels=[-5, -3, 0, 3], n_train=800):
    """Train model on multiple SNR levels"""
    print(f"  Training on SNR levels: {snr_levels}")
    
    X_train, y_train = [], []
    
    for snr in snr_levels:
        for i in range(n_train // len(snr_levels)):
            idx = np.random.randint(0, len(eeg_all))
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            emg = emg_all[idx % len(emg_all)]
            
            # Mix noise sources
            noise = 0.7 * eog + 0.3 * emg
            noisy = mix_at_snr(eeg, noise, snr)
            
            feat = create_features_enhanced(noisy, eog, emg)
            X_train.append(feat)
            y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    
    # Train Ridge
    print("  Fitting Ridge regression...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    return model, scaler

def evaluate_model(model, scaler, X_test, y_test, test_noisy, test_eog, test_emg):
    """Evaluate model"""
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    X_test = scaler.transform(X_test)
    
    preds = model.predict(X_test)
    
    # Metrics
    corr, _ = pearsonr(preds.mean(axis=0), y_test.mean(axis=0))
    mse = np.mean((preds - y_test)**2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.sqrt(np.mean(y_test**2))
    
    return corr, rrmse, preds

# ============================================================
# EVALUATION
# ============================================================
def evaluate(pred, truth):
    """Calculate metrics"""
    # Handle NaN/inf
    pred = np.nan_to_num(pred, nan=0, posinf=0, neginf=0)
    truth = np.nan_to_num(truth, nan=0, posinf=0, neginf=0)
    
    corr, _ = pearsonr(pred.mean(axis=0), truth.mean(axis=0))
    mse = np.mean((pred - truth)**2)
    rmse = np.sqrt(mse)
    rrmse = rmse / np.sqrt(np.mean(truth**2))
    return corr, rrmse

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("EEG DENOISING - ENHANCED v2")
    print("Novel Approaches: Wavelet, Adaptive, Multi-SNR")
    print("="*60)
    
    # Load data
    print("\n[1] Loading data...")
    eeg_all, eog_all, emg_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}, EMG: {emg_all.shape}")
    
    # Split data
    n_total = min(1500, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Prepare test data at -5dB (the target SNR)
    print("\n[2] Preparing test data at SNR=-5dB...")
    X_test, y_test, noisy_test, eog_test, emg_test = [], [], [], [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        emg = emg_all[idx % len(emg_all)]
        
        # Mix multiple noise sources
        noise = 0.7 * eog + 0.3 * emg
        noisy = mix_at_snr(eeg, noise, -5)
        
        X_test.append(create_features_enhanced(noisy, eog, emg))
        y_test.append(eeg)
        noisy_test.append(noisy)
        eog_test.append(eog)
        emg_test.append(emg)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    noisy_test = np.array(noisy_test, dtype=np.float32)
    eog_test = np.array(eog_test, dtype=np.float32)
    emg_test = np.array(emg_test, dtype=np.float32)
    
    print(f"    Test shape: {X_test.shape}")
    
    results = {}
    
    # ============================================================
    # METHOD 1: Bandpass (Baseline)
    # ============================================================
    print("\n[3a] Bandpass filter...")
    preds_bp = np.array([butterworth_bandpass(ns) for ns in noisy_test])
    corr, rrmse = evaluate(preds_bp, y_test)
    results['Bandpass'] = (corr, rrmse)
    print(f"    Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 2: Wiener (Baseline)
    # ============================================================
    print("\n[3b] Wiener filter...")
    preds_wiener = np.array([wiener_filter(ns) for ns in noisy_test])
    corr, rrmse = evaluate(preds_wiener, y_test)
    results['Wiener'] = (corr, rrmse)
    print(f"    Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 3: Wavelet Denoising (Novel)
    # ============================================================
    print("\n[3c] Wavelet denoising...")
    preds_wavelet = wavelet_denoise_batch(noisy_test)
    corr, rrmse = evaluate(preds_wavelet, y_test)
    results['Wavelet'] = (corr, rrmse)
    print(f"    Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 4: Adaptive Filter (LMS) (Novel)
    # ============================================================
    print("\n[3d] Adaptive filter (LMS)...")
    preds_lms = adaptive_filter_batch(noisy_test, eog_test, method='lms', mu=0.02)
    corr, rrmse = evaluate(preds_lms, y_test)
    results['LMS'] = (corr, rrmse)
    print(f"    Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 5: Single-SNR Ridge (Baseline ML)
    # ============================================================
    print("\n[3e] Ridge regression (single SNR -5dB)...")
    X_train, y_train = [], []
    for idx in train_idx[:500]:  # Smaller for speed
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        emg = emg_all[idx % len(emg_all)]
        noise = 0.7 * eog + 0.3 * emg
        noisy = mix_at_snr(eeg, noise, -5)
        X_train.append(create_features_enhanced(noisy, eog, emg))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    
    scaler1 = StandardScaler()
    X_train_s = scaler1.fit_transform(X_train)
    X_test_s = scaler1.transform(X_test)
    
    model1 = Ridge(alpha=1.0)
    model1.fit(X_train_s, y_train)
    preds_ridge = model1.predict(X_test_s)
    corr, rrmse = evaluate(preds_ridge, y_test)
    results['Ridge (-5dB)'] = (corr, rrmse)
    print(f"    Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 6: Multi-SNR Training (Novel)
    # ============================================================
    print("\n[3f] Multi-SNR training...")
    model2, scaler2 = train_multisnr_model(eeg_all, eog_all, emg_all, 
                                           snr_levels=[-5, -3, 0, 3], n_train=800)
    corr, rrmse, _ = evaluate_model(model2, scaler2, X_test, y_test, noisy_test, eog_test, emg_test)
    results['Ridge (Multi-SNR)'] = (corr, rrmse)
    print(f"    Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ============================================================
    # METHOD 7: Ensemble (Average of best methods)
    # ============================================================
    print("\n[3g] Ensemble (Wavelet + Ridge Multi-SNR)...")
    preds_ensemble = 0.5 * preds_wavelet + 0.5 * preds_ridge
    corr, rrmse = evaluate(preds_ensemble, y_test)
    results['Ensemble'] = (corr, rrmse)
    print(f"    Pearson={corr:.4f}, RRMSE={rrmse:.4f}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Pearson':>10} {'RRMSE':>10}")
    print("-"*40)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    for method, (corr, rrmse) in sorted_results:
        status = "✅" if corr >= 0.98 and rrmse <= 0.15 else ""
        print(f"{method:<20} {corr:>10.4f} {rrmse:>10.4f} {status}")
    
    print("\nTarget: Pearson ≥ 0.98, RRMSE ≤ 0.15")
    
    best_method = sorted_results[0][0]
    best_corr = sorted_results[0][1][0]
    if best_corr < 0.98:
        print(f"\n⚠️ Target not met. Best: {best_method} with Pearson={best_corr:.4f}")
        print("\nNext ideas:")
        print("- Try deeper neural networks")
        print("- Use pretrained EEG encoders")
        print("- Try different wavelet families")
    else:
        print(f"\n🎉 Target met with {best_method}!")

if __name__ == "__main__":
    main()
