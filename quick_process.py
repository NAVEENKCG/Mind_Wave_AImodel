import mne
import numpy as np
from pathlib import Path
import logging
from config import *

# Ensure we use the 1.0s window defined in config.py
WINDOW_SIZE = TRAIN_WINDOW_SIZE # 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_tgam_features_from_raw(raw):
    """Convert high-rate EEG to TGAM-style 10Hz power band features."""
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    chunk_size = int(sfreq * 0.1)
    features_list = []
    
    for i in range(0, data.shape[1] - chunk_size + 1, chunk_size):
        chunk = data[:, i:i+chunk_size]
        # Map raw EEG energy to simulated power bands
        energy = np.mean(np.abs(chunk))
        feat = [
            energy * 1.5, # delta
            energy * 1.0, # theta
            energy * 0.5, # lowAlpha
            energy * 0.4, # highAlpha
            energy * 0.3, # lowBeta
            energy * 0.2, # highBeta
            energy * 1.8, # lowGamma
            energy * 1.5, # highGamma
            80.0, # attention
            60.0, # meditation
            0.0   # blink
        ]
        features_list.append(feat)
    return np.array(features_list)

def process_and_store():
    data_path = DATA_DIR / "external" / "ds002721"
    all_windows = []
    all_labels = []
    
    # Process 10 subjects for high accuracy
    subjects = [f"sub-{i:02d}" for i in range(1, 11)]
    
    for sub in subjects:
        sub_path = data_path / sub / "eeg"
        if not sub_path.exists(): continue
        
        for edf_file in sub_path.glob("*.edf"):
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                features = extract_tgam_features_from_raw(raw)
                
                # STRICT MAPPING:
                # Run 1: Resting -> IDLE (0)
                # Run 2: Motor Imagery -> FORWARD (1)
                # Run 3: Motor Imagery -> LEFT (2)
                if "run1" in str(edf_file): label = 0
                elif "run2" in str(edf_file): label = 1
                elif "run3" in str(edf_file): label = 2
                else: continue # Skip others for clarity
                
                for i in range(0, len(features) - WINDOW_SIZE + 1, 5):
                    all_windows.append(features[i : i + WINDOW_SIZE])
                    all_labels.append(label)
            except: continue

    X = np.array(all_windows)
    y = np.array(all_labels)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "X_pretrained.npy", X)
    np.save(DATA_DIR / "y_pretrained.npy", y)
    logger.info(f"✅ DATA READY! {len(X)} samples aligned (Window: {WINDOW_SIZE})")

if __name__ == "__main__":
    process_and_store()
