#!/usr/bin/env python3
"""
Per-Subject EEG Denoising - Fixed Version
"""

import numpy as np
from scipy import signal, stats
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

try:
    from sklearn.decomposition import FastICA
    HAS_ICA = True
except ImportError:
    HAS_ICA = False

print("="*70)
print("PER-SUBJECT EEG DENOISING EXPERIMENT v2")
print("="*70)

# Load data
print("\n[1] Loading data...")
data_dir = '/Users/lobter/.openclaw/workspace/EEGdenoiseNet-Pipeline/data/'
eeg_full = np.load(data_dir + 'EEG_all_epochs.npy')
emg_full = np.load(data_dir + 'EMG_all_epochs.npy')
eog_full = np.load(data_dir + 'EOG_all_epochs.npy')

n_epochs = min(len(eeg_full), len(emg_full), len(eog_full))
eeg_clean = eeg_full[:n_epochs]
emg_noise = emg_full[:n_epochs]
eog_noise = eog_full[:n_epochs]

print(f"  Using {n_epochs} epochs")

# Create subject labels (simulated)
print("\n[2] Creating simulated subjects...")
n_subjects = 50  # Fewer subjects = more epochs per subject
epochs_per_subject = n_epochs // n_subjects
subject_ids = np.repeat(np.arange(n_subjects), epochs_per_subject)
if len(subject_ids) < n_epochs:
    subject_ids = np.concatenate([subject_ids, np.arange(n_epochs - len(subject_ids))])

print(f"  Subjects: {n_subjects}, epochs/subject: ~{epochs_per_subject}")

# Create noisy EEG
print("\n[3] Creating noisy EEG mixtures...")
np.random.seed(42)
noise_weights = np.random.uniform(0.3, 0.7, n_epochs).reshape(-1, 1)
mixed_noise = noise_weights * emg_noise + (1 - noise_weights) * eog_noise

def add_noise_at_snr(clean, noise, snr_db):
    signal_power = np.mean(clean**2)
    noise_power = np.mean(noise**2)
    target_noise_power = signal_power / (10**(snr_db/10))
    scale = np.sqrt(target_noise_power / (noise_power + 1e-10))
    return clean + noise * scale, scale

snr_levels = [-5, -2, 0, 2, 5]
noisy_data = {}
for snr in snr_levels:
    noisy_eeg = [add_noise_at_snr(eeg_clean[i], mixed_noise[i], snr)[0] for i in range(n_epochs)]
    noisy_data[snr] = np.array(noisy_eeg)

# Feature extraction
print("\n[4] Extracting EEG features...")

def extract_features(x, fs=256):
    # Statistical
    feats = [np.mean(x), np.std(x), np.min(x), np.max(x), np.median(x),
             stats.kurtosis(x), stats.skew(x), np.percentile(x, 25), np.percentile(x, 75),
             np.sqrt(np.mean(x**2))]
    
    # Frequency
    try:
        freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
        bands = {'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 100)}
        for _, (low, high) in bands.items():
            idx = np.logical_and(freqs >= low, freqs <= high)
            feats.append(np.sum(psd[idx]))
        # SEF95
        cumsum = np.cumsum(psd)
        sef_idx = np.searchsorted(cumsum, 0.95 * cumsum[-1])
        feats.append(freqs[min(sef_idx, len(freqs)-1)])
    except:
        feats.extend([0] * 6)
    
    # Wavelet
    if HAS_PYWT:
        try:
            coeffs = pywt.wavedec(x, 'db4', level=3)
            for c in coeffs:
                feats.extend([np.mean(np.abs(c)), np.std(c)])
        except:
            feats.extend([0] * 8)
    
    return np.array(feats)

# Extract features
X_features = np.array([extract_features(noisy_data[0][i]) for i in range(n_epochs)])
print(f"  Features shape: {X_features.shape}")

# Split: leave 4 subjects out for testing
print("\n[5] Splitting data...")
unique_subjects = np.unique(subject_ids)
np.random.seed(42)
test_subjects = np.random.choice(unique_subjects, size=10, replace=False)
train_subjects = np.setdiff1d(unique_subjects, test_subjects)

train_mask = np.isin(subject_ids, train_subjects)
test_mask = np.isin(subject_ids, test_subjects)

print(f"  Train subjects: {len(train_subjects)}, epochs: {train_mask.sum()}")
print(f"  Test subjects: {len(test_subjects)}, epochs: {test_mask.sum()}")

# Train/val split within training subjects
train_subj_arr = subject_ids[train_mask]
train_subj_unique = np.unique(train_subj_arr)
val_subjects = np.random.choice(train_subj_unique, size=8, replace=False)
train_only_subjects = np.setdiff1d(train_subj_unique, val_subjects)

train_only_mask = train_mask & np.isin(subject_ids, train_only_subjects)
val_mask = train_mask & np.isin(subject_ids, val_subjects)

print(f"  Train-only epochs: {train_only_mask.sum()}, Val epochs: {val_mask.sum()}")

# =============================================================================
# UNIVERSAL MODEL
# =============================================================================
print("\n[6] Training UNIVERSAL model...")

X_train = X_features[train_only_mask]
y_train = eeg_clean[train_only_mask]
X_val = X_features[val_mask]
y_val = eeg_clean[val_mask]
X_test = X_features[test_mask]
y_test = eeg_clean[test_mask]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

universal_model = Ridge(alpha=1.0)
universal_model.fit(X_train_s, y_train)

def calc_metrics(y_true, y_pred):
    rrmse = np.sqrt(np.mean((y_true - y_pred)**2)) / (np.sqrt(np.mean(y_true**2)) + 1e-10)
    pearson = np.mean([np.corrcoef(y_true[i], y_pred[i])[0,1] for i in range(len(y_true))])
    return rrmse, pearson

universal_pred = universal_model.predict(X_test_s)
universal_rrmse, universal_pearson = calc_metrics(y_test, universal_pred)
print(f"  Universal RRMSE: {universal_rrmse:.4f}, Pearson: {universal_pearson:.4f}")

# =============================================================================
# PER-SUBJECT MODELS
# =============================================================================
print("\n[7] Training PER-SUBJECT models...")

per_subject_preds = []
per_subject_targets = []

# For each test subject, train a model using ONLY their training data
for subj in test_subjects:
    subj_train_mask = (subject_ids == subj) & train_only_mask
    subj_test_mask = (subject_ids == subj) & test_mask
    
    n_subj_train = subj_train_mask.sum()
    n_subj_test = subj_test_mask.sum()
    
    if n_subj_train < 5 or n_subj_test == 0:
        continue
    
    X_subj_train = X_features[subj_train_mask]
    y_subj_train = eeg_clean[subj_train_mask]
    X_subj_test = X_features[subj_test_mask]
    y_subj_test = eeg_clean[subj_test_mask]
    
    # Standardize
    subj_scaler = StandardScaler()
    X_subj_train_s = subj_scaler.fit_transform(X_subj_train)
    X_subj_test_s = subj_scaler.transform(X_subj_test)
    
    # Train
    subj_model = Ridge(alpha=1.0)
    subj_model.fit(X_subj_train_s, y_subj_train)
    
    # Predict
    subj_pred = subj_model.predict(X_subj_test_s)
    
    per_subject_preds.append(subj_pred)
    per_subject_targets.append(y_subj_test)
    
    r, p = calc_metrics(y_subj_test, subj_pred)
    print(f"    Subject {subj}: train={n_subj_train}, test={n_subj_test}, R={p:.4f}")

if len(per_subject_preds) > 0:
    per_subject_preds = np.vstack(per_subject_preds)
    per_subject_targets = np.vstack(per_subject_targets)
    per_rrmse, per_pearson = calc_metrics(per_subject_targets, per_subject_preds)
    print(f"\n  Per-Subject Avg: RRMSE: {per_rrmse:.4f}, Pearson: {per_pearson:.4f}")
else:
    per_rrmse, per_pearson = universal_rrmse, universal_pearson
    print("  No valid per-subject models")

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\n  UNIVERSAL Model:     RRMSE={universal_rrmse:.4f}, Pearson={universal_pearson:.4f}")
print(f"  PER-SUBJECT Model:  RRMSE={per_rrmse:.4f}, Pearson={per_pearson:.4f}")

improvement_rrmse = (universal_rrmse - per_rrmse) / universal_rrmse * 100
improvement_pearson = (per_pearson - universal_pearson) / (universal_pearson + 1e-10) * 100

print(f"\n  Improvement: RRMSE {improvement_rrmse:+.2f}%, Pearson {improvement_pearson:+.2f}%")

if per_rrmse < universal_rrmse and per_pearson > universal_pearson:
    print("\n  ✓ PER-SUBJECT OUTPERFORMS UNIVERSAL!")
elif per_pearson > universal_pearson:
    print("\n  ~ Per-subject better on correlation")
else:
    print("\n  ✗ Universal better")

# =============================================================================
# SNR TESTING
# =============================================================================
print("\n[8] Testing at different SNR levels...")

for snr in snr_levels:
    X_snr = np.array([extract_features(noisy_data[snr][i]) for i in range(n_epochs)])
    
    X_test_snr = X_snr[test_mask]
    y_test_snr = eeg_clean[test_mask]
    X_test_snr_s = scaler.transform(X_test_snr)
    
    uni_pred = universal_model.predict(X_test_snr_s)
    uni_r, uni_p = calc_metrics(y_test_snr, uni_pred)
    
    # Per-subject for this SNR
    per_preds = []
    per_tgts = []
    for subj in test_subjects:
        subj_train_mask = (subject_ids == subj) & train_only_mask
        subj_test_mask = (subject_ids == subj) & test_mask
        if subj_train_mask.sum() < 5 or subj_test_mask.sum() == 0:
            continue
        X_st = X_features[subject_ids == subj]
        y_st = eeg_clean[subject_ids == subj]
        
        # Get this SNR test data
        X_snr_subj = X_snr[subj_test_mask]
        y_subj = eeg_clean[subj_test_mask]
        
        # Train on original features (but could also use SNR-specific)
        subj_scaler = StandardScaler()
        X_st_s = subj_scaler.fit_transform(X_st[train_only_mask[subject_ids == subj]])
        X_snr_subj_s = subj_scaler.transform(X_snr_subj)
        
        model = Ridge(alpha=1.0)
        model.fit(X_st_s, y_st[train_only_mask[subject_ids == subj]])
        pred = model.predict(X_snr_subj_s)
        
        per_preds.append(pred)
        per_tgts.append(y_subj)
    
    if per_preds:
        per_preds = np.vstack(per_preds)
        per_tgts = np.vstack(per_tgts)
        per_r, per_p = calc_metrics(per_tgts, per_preds)
    else:
        per_r, per_p = uni_r, uni_p
    
    winner = "Per-Sub" if per_p > uni_p else "Universal"
    print(f"  SNR={snr:+3d}dB: Universal R={uni_p:.4f}, Per-Sub R={per_p:.4f} -> {winner}")

# =============================================================================
# ICA
# =============================================================================
print("\n[9] ICA-based denoising...")

if HAS_ICA:
    ica_idx = np.random.choice(n_epochs, min(500, n_epochs), replace=False)
    X_ica = noisy_data[0][ica_idx]
    
    try:
        ica = FastICA(n_components=10, random_state=42, max_iter=500)
        sources = ica.fit_transform(X_ica)
        
        comp_kurt = [stats.kurtosis(sources[:, i]) for i in range(sources.shape[1])]
        artifact_comps = [i for i, k in enumerate(comp_kurt) if abs(k) > 3]
        
        sources_clean = sources.copy()
        sources_clean[:, artifact_comps] = 0
        X_recon = ica.inverse_transform(sources_clean)
        
        ica_r, ica_p = calc_metrics(eeg_clean[ica_idx], X_recon)
        print(f"  ICA: RRMSE={ica_r:.4f}, Pearson={ica_p:.4f}")
    except Exception as e:
        print(f"  ICA failed: {e}")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"""
Dataset: {n_epochs} epochs, {n_subjects} simulated subjects (~{epochs_per_subject}/subject)

Feature Types: Statistical({10}), Frequency({6}), Wavelet({8 if HAS_PYWT else 0})

Results:
  UNIVERSAL:   RRMSE={universal_rrmse:.4f}, Pearson={universal_pearson:.4f}
  PER-SUBJECT: RRMSE={per_rrmse:.4f}, Pearson={per_pearson:.4f}
  
  Improvement: RRMSE {improvement_rrmse:+.2f}%, Pearson {improvement_pearson:+.2f}%

Conclusion: Per-subject models {'outperform' if per_pearson > universal_pearson else 'do not outperform'} universal model.
""")
