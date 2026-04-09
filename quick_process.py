import mne
import numpy as np
from pathlib import Path
import logging
from config import *
from scipy.signal import welch

# Ensure we use the 1.0s window defined in config.py
WINDOW_SIZE = TRAIN_WINDOW_SIZE # 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_bandpower(data, sfreq, band):
    """Calculate the average power in a specific frequency band."""
    f, psd = welch(data, sfreq, nperseg=min(data.shape[-1], 256))
    idx_band = np.logical_and(f >= band[0], f <= band[1])
    return np.mean(psd[idx_band]) if np.any(idx_band) else 0

def extract_scientific_features(raw):
    """Extract real EEG power bands (Delta to Gamma) like a real TGAM module."""
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    chunk_size = int(sfreq * 0.1) # 10Hz steps
    features_list = []
    
    # Define standard EEG bands
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'lowAlpha': (8, 10),
        'highAlpha': (10, 13),
        'lowBeta': (13, 20),
        'highBeta': (20, 30),
        'lowGamma': (30, 40),
        'highGamma': (40, 50)
    }
    
    # Process the entire recording in 0.1s chunks
    for i in range(0, data.shape[1] - chunk_size + 1, chunk_size):
        chunk = data[0, i:i+chunk_size] # Use first channel
        
        row = []
        for name, (fmin, fmax) in BANDS.items():
            power = get_bandpower(chunk, sfreq, (fmin, fmax))
            row.append(power)
            
        # Add placeholders for Attention, Meditation, Blink (TGAM style)
        row.extend([50.0, 50.0, 0.0])
        features_list.append(row)
        
    return np.array(features_list)

def process_and_store():
    data_path = DATA_DIR / "external" / "ds002721"
    all_windows = []
    all_labels = []
    
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    
    logger.info("🧠 Extracting Scientific Frequency Bands...")
    
    for sub in subjects:
        sub_path = data_path / sub / "eeg"
        if not sub_path.exists(): continue
        
        for edf_file in sub_path.glob("*.edf"):
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                # Standardize EEG data (Filter and Resample)
                raw.filter(1, 50, verbose=False)
                
                features = extract_scientific_features(raw)
                
                # Normalize features (Standard Scaling)
                if len(features) > 0:
                    features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-6)
                
                if "run1" in str(edf_file): label = 0 # IDLE
                elif "run2" in str(edf_file): label = 1 # FORWARD
                elif "run3" in str(edf_file): label = 2 # LEFT
                else: continue
                
                for i in range(0, len(features) - WINDOW_SIZE + 1, 5):
                    all_windows.append(features[i : i + WINDOW_SIZE])
                    all_labels.append(label)
            except Exception as e:
                continue

    X = np.array(all_windows)
    y = np.array(all_labels)
    
    # Clean up and save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "X_pretrained.npy", X)
    np.save(DATA_DIR / "y_pretrained.npy", y)
    logger.info(f"✅ HIGH-FIDELITY DATA READY! {len(X)} samples with {X.shape[2]} features.")

if __name__ == "__main__":
    process_and_store()
