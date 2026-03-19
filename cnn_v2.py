#!/usr/bin/env python3
"""
EEG Denoising - Lightweight 1D CNN
Uses curriculum learning: train at easier SNRs
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
torch.manual_seed(42)

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
# FEATURE EXTRACTION (in Python, apply to noisy signal)
# ============================================================
def butter_bandpass(data, fs=256):
    b, a = butter(4, [0.5/128, 40/128], btype='band')
    return filtfilt(b, a, data)

# ============================================================
# LIGHTWEIGHT 1D CNN
# ============================================================
class EEGDenoiserCNN(nn.Module):
    def __init__(self, input_len=512):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2),
            nn.Conv1d(32, 1, kernel_size=7, padding=3),
        )
        
    def forward(self, x):
        # x: (batch, 512)
        x = x.unsqueeze(1)  # (batch, 1, 512)
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out.squeeze(1)  # (batch, 512)

# ============================================================
# TRAINING
# ============================================================
def train_model(model, train_loader, epochs=10, lr=1e-3, device='cpu'):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    
    for epoch in range(epochs):
        total_loss = 0
        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        if (epoch + 1) % 3 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model

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
    print("EEG DENOISING - LIGHTWEIGHT 1D CNN")
    print("="*60)
    
    device = torch.device('cpu')  # Force CPU
    print(f"Using device: {device}")
    
    # Load data
    print("\n[1] Loading data...")
    eeg_all, eog_all = load_data()
    print(f"    EEG: {eeg_all.shape}, EOG: {eog_all.shape}")
    
    # Split
    n_total = min(2000, len(eeg_all))
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42)
    print(f"    Train: {len(train_idx)}, Test: {len(test_idx)}")
    
    # Prepare test data at -5dB
    print("\n[2] Preparing test data at SNR=-5dB...")
    noisy_test, y_test = [], []
    
    for idx in test_idx:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        noisy_test.append(butter_bandpass(noisy))  # Pre-filter
        y_test.append(eeg)
    
    noisy_test = np.array(noisy_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.float32)
    print(f"    Test shape: {noisy_test.shape}")
    
    results = {}
    
    # ============================================================
    # BASELINE: No denoising (just bandpass)
    # ============================================================
    print("\n[3a] Baseline (bandpass only)...")
    corr, corr_std, rrmse, corr_avg = evaluate(noisy_test, y_test)
    results['Bandpass only'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # CNN: Train at -5dB
    # ============================================================
    print("\n[3b] CNN (train at -5dB)...")
    X_train, y_train = [], []
    for idx in train_idx[:800]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, -5)
        X_train.append(butter_bandpass(noisy))
        y_train.append(eeg)
    
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    
    # Create DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Train
    model = EEGDenoiserCNN().to(device)
    model = train_model(model, train_loader, epochs=15)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(noisy_test).to(device)
        preds_cnn = model(test_tensor).cpu().numpy()
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_cnn, y_test)
    results['CNN (-5dB)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # CNN: Curriculum (train at 0dB)
    # ============================================================
    print("\n[3c] CNN (curriculum: train at 0dB)...")
    X_train_c, y_train_c = [], []
    for idx in train_idx[:1000]:
        eeg = eeg_all[idx]
        eog = eog_all[idx % len(eog_all)]
        noisy = mix_at_snr(eeg, eog, 0)  # Easier SNR!
        X_train_c.append(butter_bandpass(noisy))
        y_train_c.append(eeg)
    
    X_train_c = np.array(X_train_c, dtype=np.float32)
    y_train_c = np.array(y_train_c, dtype=np.float32)
    
    train_dataset_c = TensorDataset(torch.FloatTensor(X_train_c), torch.FloatTensor(y_train_c))
    train_loader_c = DataLoader(train_dataset_c, batch_size=64, shuffle=True)
    
    model_c = EEGDenoiserCNN().to(device)
    model_c = train_model(model_c, train_loader_c, epochs=15, device=device)
    
    model_c.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(noisy_test).to(device)
        preds_cnn_c = model_c(test_tensor).cpu().numpy()
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_cnn_c, y_test)
    results['CNN (0dB train)'] = (corr, corr_std, rrmse, corr_avg)
    print(f"    Pearson={corr:.4f} (avg={corr_avg:.4f}), RRMSE={rrmse:.4f}")
    
    # ============================================================
    # CNN: Multi-SNR curriculum
    # ============================================================
    print("\n[3d] CNN (multi-SNR curriculum)...")
    X_train_m, y_train_m = [], []
    for snr in [-3, 0, 3]:
        for idx in train_idx[:400]:
            eeg = eeg_all[idx]
            eog = eog_all[idx % len(eog_all)]
            noisy = mix_at_snr(eeg, eog, snr)
            X_train_m.append(butter_bandpass(noisy))
            y_train_m.append(eeg)
    
    X_train_m = np.array(X_train_m, dtype=np.float32)
    y_train_m = np.array(y_train_m, dtype=np.float32)
    
    train_dataset_m = TensorDataset(torch.FloatTensor(X_train_m), torch.FloatTensor(y_train_m))
    train_loader_m = DataLoader(train_dataset_m, batch_size=64, shuffle=True)
    
    model_m = EEGDenoiserCNN().to(device)
    model_m = train_model(model_m, train_loader_m, epochs=15, device=device)
    
    model_m.eval()
    with torch.no_grad():
        test_tensor = torch.FloatTensor(noisy_test).to(device)
        preds_cnn_m = model_m(test_tensor).cpu().numpy()
    
    corr, corr_std, rrmse, corr_avg = evaluate(preds_cnn_m, y_test)
    results['CNN (Multi-SNR)'] = (corr, corr_std, rrmse, corr_avg)
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

if __name__ == "__main__":
    main()
