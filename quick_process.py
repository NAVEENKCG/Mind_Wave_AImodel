import mne
import numpy as np
from pathlib import Path
import logging
from config import *
from scipy.signal import welch

WINDOW_SIZE = TRAIN_WINDOW_SIZE  # 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard EEG frequency bands (Hz)
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


def bandpower(data_1d, sfreq, fmin, fmax):
    """Compute average power in a frequency band using Welch's method."""
    nperseg = min(len(data_1d), int(sfreq * 2))  # 2-second windows for resolution
    if nperseg < 8:
        return 0.0
    f, psd = welch(data_1d, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    idx = np.logical_and(f >= fmin, f <= fmax)
    if not np.any(idx):
        return 0.0
    return float(np.trapz(psd[idx], f[idx]))  # Integrate power (more accurate)


def extract_features(raw):
    """
    Extract TGAM-style features from multi-channel clinical EEG.
    
    Uses 1-second windows (not 0.1s) for proper frequency resolution,
    then averages across all EEG channels for robustness.
    Returns one feature row per second of data.
    """
    data = raw.get_data()          # (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    n_channels, n_samples = data.shape
    chunk_size = int(sfreq * 1.0)  # 1-second chunks for proper PSD
    rows = []

    for start in range(0, n_samples - chunk_size + 1, chunk_size):
        chunk = data[:, start : start + chunk_size]  # (n_channels, chunk_size)

        # Compute bandpower for EVERY channel, then average
        band_powers = []
        for fmin, fmax in BANDS:
            ch_powers = [bandpower(chunk[ch], sfreq, fmin, fmax) for ch in range(n_channels)]
            band_powers.append(float(np.mean(ch_powers)))

        # Derived ratio features (highly discriminative for BCI)
        epsilon = 1e-12
        alpha_total = band_powers[2] + band_powers[3] + epsilon
        beta_total  = band_powers[4] + band_powers[5] + epsilon
        theta       = band_powers[1] + epsilon

        attention  = float(np.clip(beta_total / alpha_total * 50, 0, 100))
        meditation = float(np.clip(alpha_total / theta * 50, 0, 100))
        blink = 0.0

        row = band_powers + [attention, meditation, blink]  # 8 + 3 = 11 features
        rows.append(row)

    return np.array(rows) if rows else np.empty((0, 11))


def process_and_store():
    data_path = DATA_DIR / "external" / "ds002721"
    if not data_path.exists():
        logger.error("Dataset not found. Run fetch_and_process_openneuro.py first.")
        return

    all_windows = []
    all_labels  = []

    subjects = [f"sub-{i:02d}" for i in range(1, 24)]  # Use ALL 23 subjects

    logger.info("🧠 Extracting scientific features with multi-channel PSD...")

    for sub in subjects:
        sub_path = data_path / sub / "eeg"
        if not sub_path.exists():
            continue

        for edf_file in sorted(sub_path.glob("*.edf")):
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
                raw.filter(1.0, 50.0, verbose=False)  # Bandpass filter

                features = extract_features(raw)
                if len(features) < WINDOW_SIZE:
                    continue

                # Label mapping based on run type
                fname = edf_file.stem.lower()
                if "run1" in fname or "run6" in fname:
                    label = 0   # IDLE  (resting-state runs)
                elif "run2" in fname or "run3" in fname:
                    label = 1   # FORWARD (first task block)
                elif "run4" in fname or "run5" in fname:
                    label = 2   # LEFT (second task block)
                else:
                    continue

                # Sliding window
                for i in range(0, len(features) - WINDOW_SIZE + 1, WINDOW_SIZE // 2):
                    all_windows.append(features[i : i + WINDOW_SIZE])
                    all_labels.append(label)

            except Exception as e:
                logger.debug(f"Skipped {edf_file.name}: {e}")
                continue

    if not all_windows:
        logger.error("No data was extracted. Check your dataset.")
        return

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int64)

    # Global z-score normalization (per feature across entire dataset)
    mean = X.mean(axis=(0, 1), keepdims=True)
    std  = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = (X - mean) / std

    # Save normalization stats for inference
    np.save(DATA_DIR / "norm_mean.npy", mean.squeeze())
    np.save(DATA_DIR / "norm_std.npy", std.squeeze())

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "X_pretrained.npy", X)
    np.save(DATA_DIR / "y_pretrained.npy", y)

    unique, counts = np.unique(y, return_counts=True)
    logger.info(f"✅ HIGH-FIDELITY DATA READY!")
    logger.info(f"   Shape: {X.shape}  |  Features: {X.shape[2]}")
    for cls, cnt in zip(unique, counts):
        logger.info(f"   Class {cls}: {cnt} samples")


if __name__ == "__main__":
    process_and_store()
