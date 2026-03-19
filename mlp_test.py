#!/usr/bin/env python3
"""
EEG Denoising - MLP Neural Network
Using sklearn's MLPRegressor
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# DATA
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

def butter_bandpass(data):
    b, a = butter(4, [0.5/128, 40/128], btype='band')
    return filtfilt(b, a, data)

def evaluate(pred, truth):
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

# ============================================================
# MULTI-OUTPUT WRAPPER
# ============================================================
class MultiOutputMLP:
    def __init__(self, hidden_layer_sizes=(256, 128), max_iter=100):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.models = []
        self.n_outputs = 512
        
    def fit(self, X, y):
        print(f"    Training MLP for {self.n_outputs} outputs...")
        # Train on subset of outputs for speed
        step = 16  # Predict every 16th point
        outputs = list(range(0, self.n_outputs, step))
        self.models = []
        
        for i, out_idx in enumerate(outputs):
            if i % 8 == 0:
                print(f"      Progress: {i}/{len(outputs)}")
            mlp = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                alpha=0.001,
                learning_rate_init=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                random_state=42,
                verbose=False
            )
            mlp.fit(X, y[:, out_idx])
            self.models.append((out_idx, mlp))
        return self
    
    def predict(self, X):
        results = []
        for out_idx, mlp in self.models:
            results.append(mlp.predict(X))
        
        # Reconstruct full signal via interpolation
        preds = np.zeros((X.shape[0], self.n_outputs))
        for i, (out_idx, _) in enumerate(self.models):
            if i == 0:
                for j in range(out_idx):
                    preds[:, j] = results[i][0]
            preds[:, out_idx] = results[i]
            if i < len(results) - 1:
                next_idx = self.models[i+1][0]
                for j in range(out_idx, min(next_idx, self.n_outputs)):
                    t = (j - out_idx) / (next_idx - out_idx) if next_idx > out_idx else 0
                    preds[:, j] = (1-t) * results[i][0] + t * results[i+1][0]
        
        # Fill remainder
        last_idx = self.models[-1][0]
        for j in range(last_idx, self.n_outputs):
            preds[:, j] = results[-1][0]
            
        return preds

# ============================================================
# MAIN
# ============================================================
def main():
    print("="*60)
    print("EEG DENOISING - MLP NEURAL NETWORK")
    print("="*60)
    
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    n_total = min(1500, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    print("\n[2] Preparing test data at SNR=-5dB...")
    X_test, y_test, noisy_test = [], [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_f = butter_bandpass(noisy)
        X_test.append(create_features(noisy_f, eog))
        y_test.append(eeg)
        noisy_test.append(noisy_f)
    
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    noisy_test = np.array(noisy_test, dtype=np.float32)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
    print(f"    Test shape: {X_test.shape}")
    
    results = {}
    
    # Baseline
    print("\n[3a] Bandpass baseline...")
    corr, corr_std, rrmse, corr_avg = evaluate(noisy_test, y_test)
    results['Bandpass'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f})")
    
    # Ridge
    print("\n[3b] Ridge (curriculum 0dB)...")
    X_train, y_train = [], []
    for idx in train_idx[:800]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, 0)
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
    results['Ridge'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # MLP - train on smaller feature set for speed
    print("\n[3c] MLP (curriculum)...")
    print("    Reducing feature dimensions...")
    
    # Simpler features for MLP
    def create_simple_features(x, eog_ref):
        feats = [x]
        for low, high in [(1, 30), (5, 20), (8, 13)]:
            b, a = butter(3, [low/128, high/128], btype='band')
            feats.append(filtfilt(b, a, x))
        for alpha in [0.5, 1.0, 1.5]:
            feats.append(x - alpha * eog_ref)
        return np.concatenate(feats)
    
    X_train_simple, y_train_simple = [], []
    for idx in train_idx[:600]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, 0)
        noisy_f = butter_bandpass(noisy)
        X_train_simple.append(create_simple_features(noisy_f, eog))
        y_train_simple.append(eeg)
    
    X_train_simple = np.array(X_train_simple, dtype=np.float32)
    y_train_simple = np.array(y_train_simple, dtype=np.float32)
    X_train_simple = np.nan_to_num(X_train_simple, nan=0, posinf=0, neginf=0)
    
    X_test_simple = []
    for i in range(len(X_test)):
        X_test_simple.append(create_simple_features(noisy_test[i], eog_all[test_idx[i] % len(eog_all)]))
    X_test_simple = np.array(X_test_simple, dtype=np.float32)
    X_test_simple = np.nan_to_num(X_test_simple, nan=0, posinf=0, neginf=0)
    
    scaler_simple = StandardScaler()
    X_train_simple_s = scaler_simple.fit_transform(X_train_simple)
    X_test_simple_s = scaler_simple.transform(X_test_simple)
    
    print("    Training MLP...")
    mlp = MLPRegressor(
        hidden_layer_sizes=(512, 256),
        max_iter=50,
        alpha=0.01,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        random_state=42,
        verbose=False
    )
    mlp.fit(X_train_simple_s, y_train_simple)
    preds_mlp = mlp.predict(X_test_simple_s)
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_mlp, y_test)
    results['MLP'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # Ensemble
    print("\n[3d] Ensemble (Ridge + MLP)...")
    preds_ens = 0.5 * preds_ridge + 0.5 * preds_mlp
    corr, corr_std, rrmse, corr_avg = evaluate(preds_ens, y_test)
    results['Ensemble'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # Summary
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
