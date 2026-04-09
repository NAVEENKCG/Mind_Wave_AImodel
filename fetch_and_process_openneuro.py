import mne
import numpy as np
import pandas as pd
import openneuro
import logging
from pathlib import Path
from typing import List, Dict
from config import DATA_DIR, N_FEATURES_RAW, SEED

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TGAM specific band definitions
BANDS = {
    'delta': (0.5, 3),
    'theta': (4, 7),
    'lowAlpha': (8, 10),
    'highAlpha': (11, 12),
    'lowBeta': (13, 17),
    'highBeta': (18, 29),
    'lowGamma': (30, 40),
    'highGamma': (41, 50)
}

def fetch_openneuro_dataset(dataset_id: str = "ds003478"):
    """
    Downloads a specific OpenNeuro dataset using the openneuro-py client.
    Note: Requires significant disk space and high bandwidth.
    """
    download_dir = DATA_DIR / "external" / dataset_id
    download_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Initiating download for {dataset_id} to {download_dir}...")
    try:
        # download_dir must be a string for openneuro client
        openneuro.download(dataset=dataset_id, target_dir=str(download_dir))
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}. Ensure openneuro-py is configured.")

def extract_tgam_features_from_raw(raw: mne.io.Raw) -> np.ndarray:
    """
    Processes clinical EEG Raw object to extract TGAM-like power band features.
    """
    # 1. Select Fp1 channel (or closest equivalent)
    try:
        raw.pick_channels(['Fp1'])
    except ValueError:
        logger.warning("Fp1 not found. Attempting to find alternatives...")
        # Try some common alternatives
        for alt in ['FP1', 'EEG Fp1', 'EEG 001']:
            if alt in raw.ch_names:
                raw.pick_channels([alt])
                break
    
    # 2. Resample to 128Hz (reasonable compromise) if needed
    if raw.info['sfreq'] > 128:
        raw.resample(128)
        
    # 3. Sliding window PSD calculation (matching 1s resolution)
    # We want to output features at ~10Hz (0.1s steps) to match TGAM
    sfreq = raw.info['sfreq']
    window_duration = 1.0 # 1 second window for PSD
    step_duration = 0.1   # 0.1 second step
    
    data, times = raw.get_data(return_times=True)
    samples_per_window = int(window_duration * sfreq)
    samples_per_step = int(step_duration * sfreq)
    
    features_list = []
    
    for start in range(0, data.shape[1] - samples_per_window, samples_per_step):
        window_data = data[:, start : start + samples_per_window]
        
        # Calculate PSD
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            window_data, sfreq, fmin=0.5, fmax=51, verbose=False
        )
        
        # Extract bands
        band_powers = []
        for band, (fmin, fmax) in BANDS.items():
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_powers.append(np.mean(psds[:, idx]))
            
        # Add placeholder Attention / Meditation / Blink (since we don't have them)
        # We can simulate them or leave as 0/median
        band_powers.extend([50, 50, 0]) # attn, med, blink
        
        features_list.append(band_powers)
        
    return np.array(features_list)

def process_and_store():
    """Processes downloaded EDF/Set files and maps events to commands."""
    datasets = ["ds002721", "ds003478"]
    
    all_windows = []
    all_labels = []

    for dataset_id in datasets:
        input_dir = DATA_DIR / "external" / dataset_id
        if not input_dir.exists():
            continue

        edf_files = list(input_dir.glob("**/*.edf"))
        for file_path in edf_files:
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True)
                logger.info(f"Processing {file_path.name} from {dataset_id}...")
                
                # Find events
                events_path = file_path.with_name(file_path.name.replace("_eeg.edf", "_events.tsv"))
                if not events_path.exists(): continue
                
                events_df = pd.read_csv(events_path, sep='\t')
                
                for _, event in events_df.iterrows():
                    onset, duration = event['onset'], event['duration']
                    trial_type = str(event['trial_type']).lower()
                    
                    label = None
                    if dataset_id == "ds002721":
                        if 'arithmetic' in trial_type: label = 2 # LEFT
                        elif 'rest' in trial_type: label = 0 # IDLE
                    elif dataset_id == "ds003478":
                        if 'task' in trial_type or 'focus' in trial_type: label = 1 # FORWARD
                        elif 'rest' in trial_type: label = 0 # IDLE
                    
                    if label is None: continue
                    
                    # Crop and extract
                    segment = raw.copy().crop(tmin=onset, tmax=onset + duration)
                    features = extract_tgam_features_from_raw(segment)
                    
                    for i in range(0, len(features) - 30 + 1, 10):
                        all_windows.append(features[i : i + 30])
                        all_labels.append(label)
                        
            except Exception as e:
                logger.error(f"Error in {file_path.name}: {e}")

    if all_windows:
        X_pretrained = np.array(all_windows)
        y_pretrained = np.array(all_labels)
        np.save(DATA_DIR / "X_pretrained.npy", X_pretrained)
        np.save(DATA_DIR / "y_pretrained.npy", y_pretrained)
        logger.info(f"Final training-ready set stored: {X_pretrained.shape}")

if __name__ == "__main__":
    # Fetch both datasets
    for ds in ["ds002721", "ds003478"]:
        fetch_openneuro_dataset(ds)
    
    process_and_store()
