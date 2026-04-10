import mne
import numpy as np
from pathlib import Path
import logging
from config import *
from scipy.signal import welch

WINDOW_SIZE = TRAIN_WINDOW_SIZE # 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard EEG frequency bands
BANDS = [
    (0.5, 4),   # delta
    (4, 8),     # theta
    (8, 10),    # lowAlpha
    (10, 13),   # highAlpha
    (13, 20),   # lowBeta
    (20, 30),   # highBeta
    (30, 45),   # lowGamma
    (45, 50),   # highGamma
]

def get_db_power(data_1d, sfreq, fmin, fmax):
    """Calculate power in Decibels."""
    nperseg = min(len(data_1d), int(sfreq * 1.0))
    if nperseg < 8: return -120.0
    f, psd = welch(data_1d, sfreq, nperseg=nperseg)
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx): return -120.0
    power = np.mean(psd[idx])
    return 10 * np.log10(power + 1e-12) # Log-scaling!

def extract_features(raw):
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    n_channels, n_samples = data.shape
    chunk_size = int(sfreq * 1.0) # 1s chunks
    features_list = []

    for start in range(0, n_samples - chunk_size + 1, chunk_size):
        chunk = data[:, start : start + chunk_size]
        
        # Calculate power for each band, averaged across channels
        band_powers = []
        for fmin, fmax in BANDS:
            ch_vals = [get_db_power(chunk[c], sfreq, fmin, fmax) for c in range(n_channels)]
            band_powers.append(np.mean(ch_vals))
            
        # Attention = Ratio of Beta to Alpha (in Log space, this is a subtraction)
        attention = np.mean(band_powers[4:6]) - np.mean(band_powers[2:4])
        meditation = np.mean(band_powers[2:4]) - band_powers[1]
        
        row = band_powers + [attention, meditation, 0.0]
        features_list.append(row)
        
    return np.array(features_list)

def process_and_store():
    data_path = DATA_DIR / "external" / "ds002721"
    all_windows, all_labels = [], []
    subjects = [f"sub-{i:02d}" for i in range(1, 11)] # 10 subjects

    logger.info("🧠 Processing with DECIBEL SCALING (High Accuracy Mode)...")
    
    for sub in subjects:
        sub_path = data_path / sub / "eeg"
        if not sub_path.exists(): continue
        for edf_file in sub_path.glob("*.edf"):
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                raw.filter(1, 45, verbose=False)
                features = extract_features(raw)
                
                # Normalize features
                if len(features) > 0:
                    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
                
                if "run1" in str(edf_file): label = 0
                elif "run2" in str(edf_file): label = 1
                elif "run3" in str(edf_file): label = 2
                else: continue
                
                for i in range(0, len(features) - WINDOW_SIZE + 1, 5):
                    all_windows.append(features[i : i + WINDOW_SIZE])
                    all_labels.append(label)
            except: continue

    X, y = np.array(all_windows), np.array(all_labels)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "X_pretrained.npy", X)
    np.save(DATA_DIR / "y_pretrained.npy", y)
    logger.info(f"✅ DATA READY! {len(X)} samples with Decibel Scaling.")

if __name__ == "__main__":
    process_and_store()
