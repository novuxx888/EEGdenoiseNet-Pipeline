#!/usr/bin/env python3
"""
Per-Subject EEG Denoising with EEG-Specific Features
=====================================================
Tests whether per-subject models outperform universal models.
Also explores EEG-specific properties: kurtosis, wavelets, ICA.
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Try to import pywt for wavelets
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False
    print("PyWavelets not available, skipping wavelet features")

# Try to import sklearn's FastICA
try:
    from sklearn.decomposition import FastICA
    HAS_ICA = True
except ImportError:
    HAS_ICA = False
    print("FastICA not available, skipping ICA features")

print("="*70)
print("PER-SUBJECT EEG DENOISING EXPERIMENT")
print("="*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] Loading data...")

data_dir = '/Users/lobter/.openclaw/workspace/EEGdenoiseNet-Pipeline/data/'
eeg_full = np.load(data_dir + 'EEG_all_epochs.npy')  # (4514, 512)
emg_full = np.load(data_dir + 'EMG_all_epochs.npy')  # (5598, 512)
eog_full = np.load(data_dir + 'EOG_all_epochs.npy')  # (3400, 512)

# Use smallest common size
n_epochs = min(len(eeg_full), len(emg_full), len(eog_full))
print(f"  Using {n_epochs} epochs (common size)")

eeg_clean = eeg_full[:n_epochs]
emg_noise = emg_full[:n_epochs]
eog_noise = eog_full[:n_epochs]

print(f"  EEG (clean): {eeg_clean.shape}")
print(f"  EMG (noise): {emg_noise.shape}")
print(f"  EOG (noise): {eog_noise.shape}")

# =============================================================================
# 2. CREATE SUBJECT LABELS (simulated since none exist)
# =============================================================================
print("\n[2] Creating simulated subject labels...")

# Use common size for all
n_subjects = 180
epochs_per_subject = n_epochs // n_subjects
subject_ids = np.repeat(np.arange(n_subjects), epochs_per_subject)
# Handle remainder
if len(subject_ids) < n_epochs:
    remaining = n_epochs - len(subject_ids)
    subject_ids = np.concatenate([subject_ids, np.arange(remaining)])

print(f"  Total epochs: {n_epochs}")
print(f"  Simulated subjects: {n_subjects}")
print(f"  Avg epochs/subject: {epochs_per_subject}")

# =============================================================================
# 3. CREATE NOISY EEG MIXTURES
# =============================================================================
print("\n[3] Creating noisy EEG mixtures...")

np.random.seed(42)

# Mix EMG and EOG noise at different ratios
noise_weights = np.random.uniform(0.3, 0.7, n_epochs).reshape(-1, 1)
mixed_noise = noise_weights * emg_noise + (1 - noise_weights) * eog_noise

# Add noise at various SNR levels
def add_noise_at_snr(clean, noise, snr_db):
    """Add noise to clean signal at specified SNR (dB)"""
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    target_noise_power = signal_power / (10**(snr_db/10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
    return clean + noise * scale, scale

# Create noisy EEG at different SNR
snr_levels = [-5, -2, 0, 2, 5]
noisy_data = {}

for snr in snr_levels:
    noisy_eeg = []
    for i in range(n_epochs):
        noisy, _ = add_noise_at_snr(eeg_clean[i], mixed_noise[i], snr)
        noisy_eeg.append(noisy)
    noisy_data[snr] = np.array(noisy_eeg)
    print(f"  SNR={snr}dB: mean amplitude = {noisy_data[snr].mean():.2f}")

# =============================================================================
# 4. EEG-SPECIFIC FEATURE EXTRACTION
# =============================================================================
print("\n[4] Extracting EEG-specific features...")

def extract_statistical_features(x):
    """Basic statistical features"""
    return np.array([
        np.mean(x),
        np.std(x),
        np.min(x),
        np.max(x),
        np.median(x),
        stats.kurtosis(x),  # Kurtosis - sensitive to artifacts
        stats.skew(x),
        np.percentile(x, 25),
        np.percentile(x, 75),
        np.sqrt(np.mean(x**2)),  # RMS
    ])

def extract_frequency_features(x, fs=256):
    """Frequency domain features"""
    freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    features = []
    for band_name, (low, high) in bands.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        features.append(np.sum(psd[idx]))
    
    # Spectral edge frequency (95%)
    cumsum = np.cumsum(psd)
    sef_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
    features.append(freqs[min(sef_idx, len(freqs)-1)])
    
    # Spectral entropy
    psd_norm = psd / (np.sum(psd) + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-10))
    features.append(spectral_entropy)
    
    return np.array(features)

def extract_wavelet_features(x, wavelet='db4', level=4):
    """Wavelet decomposition features"""
    if not HAS_PYWT:
        return np.array([])
    
    coeffs = pywt.wavedec(x, wavelet, level=level)
    
    features = []
    for coeff in coeffs:
        features.extend([
            np.mean(np.abs(coeff)),
            np.std(coeff),
            np.max(np.abs(coeff)),
            stats.kurtosis(coeff) if len(coeff) > 10 else 0,
        ])
    
    return np.array(features)

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def extract_eeg_features(x, fs=256):
    """Extract all EEG features from a single epoch"""
    try:
        b, a = butter_bandpass(0.5, 50, fs)
        x_filtered = filtfilt(b, a, x)
    except:
        x_filtered = x
    
    stat_feat = extract_statistical_features(x_filtered)
    freq_feat = extract_frequency_features(x_filtered, fs)
    
    if HAS_PYWT:
        wav_feat = extract_wavelet_features(x_filtered)
    else:
        wav_feat = np.array([])
    
    return np.concatenate([stat_feat, freq_feat, wav_feat])

# Extract features for all epochs
print("  Extracting features for all epochs...")
all_features = []
for i in range(n_epochs):
    if i % 500 == 0:
        print(f"    Progress: {i}/{n_epochs}")
    feats = extract_eeg_features(noisy_data[0][i])
    all_features.append(feats)

X_features = np.array(all_features)
print(f"  Feature matrix shape: {X_features.shape}")

# =============================================================================
# 5. SPLIT DATA BY SUBJECT
# =============================================================================
print("\n[5] Splitting data by subject for train/test...")

unique_subjects = np.unique(subject_ids)
train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)

train_mask = np.isin(subject_ids, train_subjects)
test_mask = np.isin(subject_ids, test_subjects)

print(f"  Train subjects: {len(train_subjects)}")
print(f"  Test subjects: {len(test_subjects)}")
print(f"  Train epochs: {train_mask.sum()}")
print(f"  Test epochs: {test_mask.sum()}")

# =============================================================================
# 6. TRAIN UNIVERSAL MODEL (baseline)
# =============================================================================
print("\n[6] Training UNIVERSAL model (baseline)...")

X_train = X_features[train_mask]
y_train = eeg_clean[train_mask]
X_test = X_features[test_mask]
y_test = eeg_clean[test_mask]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

universal_model = Ridge(alpha=1.0)
universal_model.fit(X_train_scaled, y_train)

universal_pred = universal_model.predict(X_test_scaled)

def calc_metrics(y_true, y_pred):
    """Calculate denoising metrics"""
    rrmse = np.sqrt(np.mean((y_true - y_pred)**2)) / (np.sqrt(np.mean(y_true**2)) + 1e-10)
    pearson = np.mean([np.corrcoef(y_true[i], y_pred[i])[0,1] for i in range(len(y_true))])
    return rrmse, pearson

universal_rrmse, universal_pearson = calc_metrics(y_test, universal_pred)
print(f"  UNIVERSAL Model Results:")
print(f"    RRMSE: {universal_rrmse:.4f}")
print(f"    Pearson: {universal_pearson:.4f}")

# =============================================================================
# 7. TRAIN PER-SUBJECT MODELS
# =============================================================================
print("\n[7] Training PER-SUBJECT models...")

min_epochs = 10
valid_train_subjects = [s for s in train_subjects if np.sum(subject_ids == s) >= min_epochs]

per_subject_results = {}
per_subject_models = {}

for subj in valid_train_subjects:
    subj_mask_train = subject_ids == subj
    subj_mask_test = subject_ids == subj
    
    if not np.any(test_mask & subj_mask_test):
        continue
    
    X_subj_train = X_features[subj_mask_train & train_mask]
    y_subj_train = eeg_clean[subj_mask_train & train_mask]
    X_subj_test = X_features[subj_mask_test & test_mask]
    y_subj_test = eeg_clean[subj_mask_test & test_mask]
    
    if len(X_subj_train) < min_epochs:
        continue
    
    subj_scaler = StandardScaler()
    X_subj_train_s = subj_scaler.fit_transform(X_subj_train)
    X_subj_test_s = subj_scaler.transform(X_subj_test)
    
    subj_model = Ridge(alpha=1.0)
    subj_model.fit(X_subj_train_s, y_subj_train)
    
    subj_pred = subj_model.predict(X_subj_test_s)
    
    rrmse, pearson = calc_metrics(y_subj_test, subj_pred)
    
    per_subject_results[subj] = {
        'rrmse': rrmse,
        'pearson': pearson,
        'n_train': len(X_subj_train),
        'n_test': len(X_subj_test)
    }
    per_subject_models[subj] = (subj_model, subj_scaler)

avg_per_subject_rrmse = np.mean([v['rrmse'] for v in per_subject_results.values()])
avg_per_subject_pearson = np.mean([v['pearson'] for v in per_subject_results.values()])

print(f"  Per-Subject Model Results (averaged over {len(per_subject_results)} subjects):")
print(f"    Avg RRMSE: {avg_per_subject_rrmse:.4f}")
print(f"    Avg Pearson: {avg_per_subject_pearson:.4f}")

# =============================================================================
# 8. COMPARE AND ANALYZE
# =============================================================================
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)

print(f"\n  UNIVERSAL Model:     RRMSE={universal_rrmse:.4f}, Pearson={universal_pearson:.4f}")
print(f"  PER-SUBJECT Model:  RRMSE={avg_per_subject_rrmse:.4f}, Pearson={avg_per_subject_pearson:.4f}")

improvement_rrmse = (universal_rrmse - avg_per_subject_rrmse) / universal_rrmse * 100
improvement_pearson = (avg_per_subject_pearson - universal_pearson) / universal_pearson * 100

print(f"\n  Improvement: RRMSE {improvement_rrmse:+.2f}%, Pearson {improvement_pearson:+.2f}%")

if avg_per_subject_rrmse < universal_rrmse and avg_per_subject_pearson > universal_pearson:
    print("\n  ✓ PER-SUBJECT models OUTPERFORM universal model!")
elif avg_per_subject_rrmse < universal_rrmse or avg_per_subject_pearson > universal_pearson:
    print("\n  ~ Mixed results - per-subject better on some metrics")
else:
    print("\n  ✗ UNIVERSAL model performs better")

# =============================================================================
# 9. TEST AT DIFFERENT SNR LEVELS
# =============================================================================
print("\n[9] Testing at different SNR levels...")

snr_comparison = []

for snr in snr_levels:
    feats_snr = []
    for i in range(n_epochs):
        feats = extract_eeg_features(noisy_data[snr][i])
        feats_snr.append(feats)
    X_snr = np.array(feats_snr)
    
    X_test_snr = X_snr[test_mask]
    y_test_snr = eeg_clean[test_mask]
    
    X_test_snr_scaled = scaler.transform(X_test_snr)
    uni_pred_snr = universal_model.predict(X_test_snr_scaled)
    uni_rrmse, uni_pearson = calc_metrics(y_test_snr, uni_pred_snr)
    
    per_subj_preds = []
    per_subj_targets = []
    
    for subj in per_subject_results.keys():
        subj_mask_test = subject_ids == subj
        X_subj_test_snr = X_snr[subj_mask_test & test_mask]
        y_subj_test = eeg_clean[subj_mask_test & test_mask]
        
        if len(X_subj_test_snr) > 0 and subj in per_subject_models:
            model, subj_scaler = per_subject_models[subj]
            X_subj_test_s = subj_scaler.transform(X_subj_test_snr)
            pred = model.predict(X_subj_test_s)
            per_subj_preds.extend(pred)
            per_subj_targets.extend(y_subj_test)
    
    if len(per_subj_preds) > 0:
        per_rrmse, per_pearson = calc_metrics(np.array(per_subj_targets), np.array(per_subj_preds))
    else:
        per_rrmse, per_pearson = uni_rrmse, uni_pearson
    
    snr_comparison.append({
        'snr': snr,
        'universal_rrmse': uni_rrmse,
        'universal_pearson': uni_pearson,
        'per_subject_rrmse': per_rrmse,
        'per_subject_pearson': per_pearson
    })
    
    print(f"  SNR={snr:+3d}dB: Universal R={uni_pearson:.4f}, Per-Subject R={per_pearson:.4f}")

# =============================================================================
# 10. ICA-BASED DENOISING (bonus)
# =============================================================================
print("\n[10] Testing ICA-based artifact removal...")

if HAS_ICA:
    ica_subset_size = min(500, n_epochs)
    ica_indices = np.random.choice(n_epochs, ica_subset_size, replace=False)
    
    X_ica = noisy_data[0][ica_indices]
    
    try:
        ica = FastICA(n_components=10, random_state=42, max_iter=500)
        sources = ica.fit_transform(X_ica)
        
        component_kurtosis = [stats.kurtosis(sources[:, i]) for i in range(sources.shape[1])]
        
        print(f"  Component kurtosis: {[f'{k:.2f}' for k in component_kurtosis]}")
        
        artifact_threshold = 3.0
        artifact_components = [i for i, k in enumerate(component_kurtosis) if abs(k) > artifact_threshold]
        
        print(f"  Identified artifact components: {artifact_components}")
        
        sources_clean = sources.copy()
        sources_clean[:, artifact_components] = 0
        
        X_ica_reconstructed = ica.inverse_transform(sources_clean)
        
        ica_pred = X_ica_reconstructed
        ica_rrmse, ica_pearson = calc_metrics(eeg_clean[ica_indices], ica_pred)
        
        print(f"  ICA denoising: RRMSE={ica_rrmse:.4f}, Pearson={ica_pearson:.4f}")
        
    except Exception as e:
        print(f"  ICA failed: {e}")
        ica_rrmse, ica_pearson = None, None
else:
    ica_rrmse, ica_pearson = None, None

# =============================================================================
# 11. FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"""
Dataset:
  - Total EEG epochs: {n_epochs}
  - Simulated subjects: {n_subjects}
  - Epochs per subject: ~{epochs_per_subject}

Feature Engineering:
  - Statistical: mean, std, min, max, median, kurtosis, skew, quartiles, RMS
  - Frequency: band powers (delta, theta, alpha, beta, gamma), SEF, spectral entropy
  - Wavelet: {'Yes (db4)' if HAS_PYWT else 'No'}
  - ICA: {'Yes' if HAS_ICA else 'No'}

Results:
  UNIVERSAL Model:     RRMSE={universal_rrmse:.4f}, Pearson={universal_pearson:.4f}
  PER-SUBJECT Model:  RRMSE={avg_per_subject_rrmse:.4f}, Pearson={avg_per_subject_pearson:.4f}
  
  Per-subject improvement: RRMSE {improvement_rrmse:+.2f}%, Pearson {improvement_pearson:+.2f}%

SNR Sensitivity:
""")

for res in snr_comparison:
    winner = "Per-Subject" if res['per_subject_pearson'] > res['universal_pearson'] else "Universal"
    print(f"  SNR={res['snr']:+3d}dB: {winner} wins (R={max(res['per_subject_pearson'], res['universal_pearson']):.4f})")

print(f"""
Key Findings:
1. Per-subject models {'outperform' if improvement_pearson > 0 else 'do not outperform'} universal model
2. The improvement is {'significant' if abs(improvement_pearson) > 5 else 'modest' if abs(improvement_pearson) > 1 else 'negligible'}
3. Subject-specific patterns exist {'and can be exploited' if improvement_pearson > 1 else 'but are minimal in this data'}
""")

print("="*70)
