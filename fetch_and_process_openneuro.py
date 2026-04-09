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
                # Memory efficient: don't preload the whole file
                raw = mne.io.read_raw_edf(file_path, preload=False)
                logger.info(f"Scanning {file_path.name}...")
                
                # Find events (handle both 'sub-01_task-run1_events.tsv' and 'sub-01_task-run1_eeg_events.tsv')
                events_name = file_path.name.replace("_eeg.edf", "_events.tsv")
                events_path = file_path.with_name(events_name)
                
                if not events_path.exists():
                    alternative_name = file_path.name.replace(".edf", "_events.tsv")
                    events_path = file_path.with_name(alternative_name)
                
                if not events_path.exists():
                    logger.warning(f"   No events found for {file_path.name}. Skipping.")
                    continue
                
                events_df = pd.read_csv(events_path, sep='\t')
                logger.info(f"   Processing {len(events_df)} events for {file_path.name}...")
                
                for idx, event in events_df.iterrows():
                    onset, duration = event['onset'], event['duration']
                    trial_type = str(event['trial_type']).lower()
                    
                    label = None
                    # Numerical mapping for many OpenNeuro datasets (0=Rest, 1=Task)
                    try:
                        t_val = int(float(trial_type))
                        if dataset_id == "ds002721":
                            if t_val == 1: label = 2 # Task -> LEFT
                            elif t_val == 0: label = 0 # Rest -> IDLE
                        elif dataset_id == "ds003478":
                            if t_val == 1: label = 1 # Task -> FORWARD
                            elif t_val == 0: label = 0 # Rest -> IDLE
                    except:
                        # Fallback for datasets with string names
                        if 'arithmetic' in trial_type: label = 2 # LEFT
                        elif 'task' in trial_type or 'focus' in trial_type: label = 1 # FORWARD
                        elif 'rest' in trial_type: label = 0 # IDLE
                    
                    if label is None: continue
                    
                    # Only load this tiny segment of data into memory
                    segment = raw.copy().crop(tmin=onset, tmax=onset + duration).load_data()
                    features = extract_tgam_features_from_raw(segment)
                    
                    # Use dynamic window size from config.py
                    for i in range(0, len(features) - TRAIN_WINDOW_SIZE + 1, 10):
                        all_windows.append(features[i : i + TRAIN_WINDOW_SIZE])
                        all_labels.append(label)
                    
                    if len(all_labels) % 100 == 0:
                        logger.info(f"   Collected {len(all_labels)} windows so far...")
                        
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
