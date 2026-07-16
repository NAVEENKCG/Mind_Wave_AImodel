"""
ORBIT AI — MOABB Training Pipeline (v6.0 — FBCSP High-Confidence)
==================================================================
CORE IMPROVEMENTS over v5.0:
  1. Filter Bank CSP (FBCSP) — extracts CSP features from 5 EEG sub-bands
     (delta, theta, alpha, beta, gamma) and concatenates them for classification.
     This exploits band-specific ERD/ERS patterns for much higher discriminability.
  2. 10 subjects × 6 motor runs + 2 rest runs = 5× more training data.
  3. Wider bandpass (0.5–50 Hz) preserves full delta and low gamma.
  4. Time-jitter augmentation (±25 samples) triples the dataset.
  5. Platt-calibrated probabilities for sharper confidence output.

Training vs Inference symmetry:
  Both classes extracted using the SAME method:
    raw.get_data() → filter(0.5,50) → chunk into 1s windows
  FORWARD class: chunks from event-marked motor imagery periods in R04/R06/R08/R10/R12/R14
  IDLE class:    chunks from continuous baseline recordings R01/R02
"""

import numpy as np
import pickle
import warnings
import urllib.request
import tempfile
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

import mne

# Paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

MODEL_OUT = MODELS_DIR / "moabb_csp_lda.pkl"
STATS_OUT = DATA_DIR   / "moabb_stats.pkl"

# ── Constants ────────────────────────────────────────────────────
SFREQ = 160           # PhysioNet sample rate
EPOCH_SAMPLES = 160   # 1 second at 160Hz (matches simulator chunk_size)
N_SUBJECTS = 10       # 10 subjects for robust covariance estimates

# ── FBCSP frequency sub-bands ───────────────────────────────────
FILTER_BANDS = [
    (0.5,  4,  "delta"),
    (4,    8,  "theta"),
    (8,   13,  "alpha"),
    (13,  30,  "beta"),
    (30,  50,  "gamma"),
]


# ══════════════════════════════════════════════════════════════════
# FBCSP Transformer — filters each epoch into sub-bands and
# extracts Riemannian tangent-space features per band.
# ══════════════════════════════════════════════════════════════════
class BandpassFilter(BaseEstimator, TransformerMixin):
    """Filter raw epochs to a specific frequency band."""
    def __init__(self, low=0.5, high=50.0, sfreq=160):
        self.low = low
        self.high = high
        self.sfreq = sfreq

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """X shape: (n_epochs, n_channels, n_times)"""
        filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            filtered[i] = mne.filter.filter_data(
                X[i].astype(np.float64), self.sfreq,
                l_freq=self.low, h_freq=self.high,
                verbose=False
            )
        return filtered


class FBCSPTransformer(BaseEstimator, TransformerMixin):
    """
    Filter Bank CSP: filters each epoch into N sub-bands,
    computes Riemannian tangent-space features per band,
    and concatenates them into a single feature vector.
    """
    def __init__(self, bands=None, sfreq=160):
        self.bands = bands or FILTER_BANDS
        self.sfreq = sfreq
        self.band_pipelines_ = []

    def fit(self, X, y=None):
        self.band_pipelines_ = []
        for low, high, name in self.bands:
            bp = BandpassFilter(low=low, high=high, sfreq=self.sfreq)
            X_band = bp.transform(X)
            cov_ts = Pipeline([
                ('cov', Covariances(estimator='lwf')),
                ('ts', TangentSpace(metric='riemann')),
            ])
            cov_ts.fit(X_band, y)
            self.band_pipelines_.append((name, bp, cov_ts))
        return self

    def transform(self, X):
        features_list = []
        for name, bp, cov_ts in self.band_pipelines_:
            X_band = bp.transform(X)
            feats = cov_ts.transform(X_band)
            features_list.append(feats)
        return np.hstack(features_list)


def download_edf(subject_id, run_id):
    """Download a specific PhysioNet EEG Motor Imagery run (with local caching)."""
    cache_dir = DATA_DIR / "edf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"S{subject_id:03d}R{run_id:02d}.edf"
    
    if cache_file.exists() and cache_file.stat().st_size > 1000:
        print(f"      Loading S{subject_id:03d}R{run_id:02d} (cached)... OK", flush=True)
        return str(cache_file)
        
    url = f"https://physionet.org/files/eegmmidb/1.0.0/S{subject_id:03d}/S{subject_id:03d}R{run_id:02d}.edf"
    print(f"      Downloading S{subject_id:03d}R{run_id:02d}...", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            with open(cache_file, "wb") as f:
                f.write(resp.read())
        print(" OK", flush=True)
        return str(cache_file)
    except Exception as e:
        print(f" FAILED ({e})", flush=True)
        if cache_file.exists():
            try: cache_file.unlink()
            except: pass
        return None


def extract_epochs_from_edf(edf_path, epoch_samples, event_aligned=False):
    """
    Extract epochs from an EDF file using the SAME method as the simulator.
    
    If event_aligned=True: extract 1s windows starting at each motor event.
    If event_aligned=False: chunk the entire recording into 1s windows (for rest).
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(0.5, 50, verbose=False)  # Wider bandpass: preserves delta floor + gamma ceiling
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels = data.shape[0]
    
    epochs = []
    
    if event_aligned:
        # Extract windows aligned to motor imagery events (T1=left, T2=right)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        motor_ids = [eid for name, eid in event_id.items() if 'T1' in name or 'T2' in name]
        
        for ev in events:
            if ev[2] not in motor_ids:
                continue
            start = ev[0]
            if start + epoch_samples > data.shape[1]:
                continue
            epochs.append(data[:, start:start + epoch_samples])
    else:
        # Chunk entire recording into fixed windows (for baseline/rest)
        n_samples = data.shape[1]
        for start in range(0, n_samples - epoch_samples, epoch_samples):
            epochs.append(data[:, start:start + epoch_samples])
    
    raw.close()
    return epochs, n_channels


def augment_epochs(epochs, max_jitter=25, rng=None):
    """
    Time-jitter augmentation: create shifted copies of each epoch.
    Adds ±jitter versions, effectively tripling the dataset.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    augmented = list(epochs)  # Keep originals
    n_times = epochs[0].shape[1] if len(epochs) > 0 else 0
    
    for epoch in epochs:
        n_ch, n_t = epoch.shape
        # Forward jitter
        jitter = rng.randint(10, max_jitter + 1)
        if jitter < n_t:
            shifted = np.zeros_like(epoch)
            shifted[:, :n_t - jitter] = epoch[:, jitter:]
            shifted[:, n_t - jitter:] = epoch[:, -1:]  # Pad with last value
            augmented.append(shifted)
        
        # Backward jitter
        jitter = rng.randint(10, max_jitter + 1)
        if jitter < n_t:
            shifted = np.zeros_like(epoch)
            shifted[:, jitter:] = epoch[:, :n_t - jitter]
            shifted[:, :jitter] = epoch[:, :1]  # Pad with first value
            augmented.append(shifted)
    
    return augmented


def main():
    print("=" * 60, flush=True)
    print("  ORBIT AI -- Training Pipeline (v6.0 — FBCSP)", flush=True)
    print("  Filter Bank CSP + wider bandpass + augmentation", flush=True)
    print("=" * 60, flush=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_epochs_path = DATA_DIR / "moabb_epochs_v6.pkl"
    
    if cache_epochs_path.exists():
        print("\n[1-3/5] Loading pre-extracted & balanced epochs from local cache...", flush=True)
        with open(cache_epochs_path, "rb") as f:
            X, y, n_channels = pickle.load(f)
        print(f"   Total epochs loaded: {len(y)} ({len(X)//2} per class)", flush=True)
        print(f"   Shape: {X.shape}", flush=True)
    else:
        subjects = list(range(1, N_SUBJECTS + 1))
        motor_runs = [4, 6, 8, 10, 12, 14]   # All left/right fist motor imagery runs
        rest_runs  = [1, 2]                    # Eyes-open + eyes-closed baselines
        
        all_motor_epochs = []
        all_rest_epochs  = []
        n_channels = None
        
        # ── 1. Extract MOTOR IMAGERY epochs ──────────────────────────────
        print(f"\n[1/5] Downloading & extracting MOTOR IMAGERY epochs...", flush=True)
        print(f"       Subjects: {len(subjects)}, Runs: {motor_runs}", flush=True)
        
        for subj in subjects:
            for run in motor_runs:
                path = download_edf(subj, run)
                if path is None:
                    continue
                try:
                    epochs, nc = extract_epochs_from_edf(path, EPOCH_SAMPLES, event_aligned=True)
                    if n_channels is None:
                        n_channels = nc
                    all_motor_epochs.extend(epochs)
                except Exception as e:
                    print(f"   [WARN] S{subj:03d}R{run:02d}: {e}", flush=True)
        
        print(f"   Extracted {len(all_motor_epochs)} motor imagery epochs", flush=True)
        
        # ── 2. Extract REST epochs ───────────────────────────────────────
        print(f"\n[2/5] Downloading & extracting REST epochs...", flush=True)
        print(f"       Subjects: {len(subjects)}, Runs: {rest_runs}", flush=True)
        
        for subj in subjects:
            for run in rest_runs:
                path = download_edf(subj, run)
                if path is None:
                    continue
                try:
                    epochs, _ = extract_epochs_from_edf(path, EPOCH_SAMPLES, event_aligned=False)
                    all_rest_epochs.extend(epochs)
                except Exception as e:
                    print(f"   [WARN] S{subj:03d}R{run:02d}: {e}", flush=True)
        
        print(f"   Extracted {len(all_rest_epochs)} rest epochs", flush=True)
        
        if len(all_motor_epochs) == 0 or len(all_rest_epochs) == 0:
            print("[ERROR] Not enough data. Aborting.", flush=True)
            return
        
        # ── 3. Augment & Balance ─────────────────────────────────────────
        print(f"\n[3/5] Augmenting & balancing dataset...", flush=True)
        
        rng = np.random.RandomState(42)
        
        print(f"   Motor epochs before augmentation: {len(all_motor_epochs)}", flush=True)
        all_motor_epochs = augment_epochs(all_motor_epochs, max_jitter=25, rng=rng)
        print(f"   Motor epochs after augmentation:  {len(all_motor_epochs)}", flush=True)
        
        print(f"   Rest epochs before augmentation:  {len(all_rest_epochs)}", flush=True)
        all_rest_epochs = augment_epochs(all_rest_epochs, max_jitter=25, rng=rng)
        print(f"   Rest epochs after augmentation:   {len(all_rest_epochs)}", flush=True)
        
        X_motor = np.array(all_motor_epochs)
        X_rest  = np.array(all_rest_epochs)
        
        n_balanced = min(len(X_motor), len(X_rest))
        
        X_motor = X_motor[rng.choice(len(X_motor), n_balanced, replace=False)]
        X_rest  = X_rest[rng.choice(len(X_rest), n_balanced, replace=False)]
        
        X = np.concatenate([X_rest, X_motor], axis=0)
        y = np.concatenate([np.zeros(n_balanced, dtype=int), np.ones(n_balanced, dtype=int)])
        
        shuffle_idx = rng.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        
        print(f"   Total epochs: {len(y)} ({n_balanced} per class)", flush=True)
        print(f"   Shape: {X.shape}", flush=True)
        
        print("\n   Caching balanced epochs to disk...", flush=True)
        with open(cache_epochs_path, "wb") as f:
            pickle.dump((X, y, n_channels), f, protocol=4)
        print("   OK saved to moabb_epochs_v6.pkl", flush=True)
        
    # ── 4. Architecture tournament ───────────────────────────────────
    print(f"\n[4/5] Architecture tournament (5-Fold CV, n_jobs=1 single process)...", flush=True)
    
    pipelines = {
        "MDM": Pipeline([
            ('cov', Covariances(estimator='lwf')),
            ('mdm', MDM(metric='riemann'))
        ]),
        "TS_LDA": Pipeline([
            ('cov', Covariances(estimator='lwf')),
            ('ts', TangentSpace(metric='riemann')),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]),
        "FBCSP_LDA": Pipeline([
            ('fbcsp', FBCSPTransformer(bands=FILTER_BANDS, sfreq=SFREQ)),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
        ]),
    }
    
    best_score = 0
    best_name = ""
    best_pipe = None
    best_scores = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, pipe in pipelines.items():
        try:
            print(f"   Evaluating {name:12} across 5 folds...", flush=True)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=1)
            mean_acc = scores.mean()
            print(f"   - {name:12}: {mean_acc*100:6.2f}% (+/- {scores.std()*100:.2f}%)", flush=True)
            if mean_acc > best_score:
                best_score = mean_acc
                best_name = name
                best_pipe = pipe
                best_scores = scores
        except Exception as e:
            print(f"   - {name:12}: FAILED ({e})", flush=True)

    print(f"\n   [WINNER] {best_name} ({best_score*100:.2f}%)", flush=True)
    pipeline = best_pipe

    # ── 5. Train final model with Platt calibration ──────────────────
    print("\n[5/5] Fitting final model with Platt calibration...", flush=True)
    
    # First fit the base pipeline
    pipeline.fit(X, y)
    
    # Wrap with Platt scaling for calibrated probabilities
    calibrated = CalibratedClassifierCV(pipeline, method='sigmoid', cv=3)
    calibrated.fit(X, y)

    # ── 6. Save ──────────────────────────────────────────────────────
    ch_mean = np.mean(X, axis=(0, 2))
    ch_std  = np.std(X,  axis=(0, 2)) + 1e-6

    uses_fbcsp = "FBCSP" in best_name

    stats = {
        'ch_mean'          : ch_mean,
        'ch_std'           : ch_std,
        'n_channels'       : n_channels,
        'n_times'          : EPOCH_SAMPLES,
        'sfreq'            : float(SFREQ),
        'uses_fbcsp'       : uses_fbcsp,
        'filter_bands'     : FILTER_BANDS if uses_fbcsp else None,
        'uses_tangent_space': False,
        'classes'          : {0: 'IDLE', 1: 'FORWARD'},
        'cv_mean'          : float(best_scores.mean()),
        'cv_std'           : float(best_scores.std()),
        'subjects'         : subjects,
        'pipeline'         : best_name,
        'calibrated'       : True,
    }

    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(calibrated, f)
    with open(STATS_OUT, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\n[SAVED] Model  -> {MODEL_OUT}", flush=True)
    print(f"[SAVED] Stats  -> {STATS_OUT}", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  CV Accuracy  : {best_scores.mean()*100:.2f}%", flush=True)
    print(f"  Subjects     : {len(subjects)}", flush=True)
    print(f"  Channels     : {n_channels}", flush=True)
    print(f"  Epoch length : {EPOCH_SAMPLES} samples (1s @ {SFREQ}Hz)", flush=True)
    print(f"  Pipeline     : {best_name} (Platt-calibrated)", flush=True)
    print(f"  Augmentation : 3× (time jitter ±25 samples)", flush=True)
    print(f"  Filter bands : {len(FILTER_BANDS)} sub-bands" if uses_fbcsp else "  Filter bands : Single broadband", flush=True)
    print(f"  Classes      : 0=IDLE, 1=FORWARD", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
