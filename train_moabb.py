"""
ORBIT AI — MOABB Training Pipeline (v7.0 — Leakage-Free FBCSP)
==================================================================
CORE IMPROVEMENTS over v6.0:
  1. FIX: Augmentation moved INSIDE CV loop to eliminate data leakage.
     The v6 pipeline augmented before splitting, allowing augmented copies
     of the same epoch into both train and test folds (inflating accuracy).
  2. Channel Selection: 64 → 22 motor-cortex channels (sensorimotor strip).
     Reduces model size from ~3.5 GB to ~200 MB and removes noise channels.
  3. Subject labels preserved: enables Leave-One-Subject-Out (LOSO) CV.
  4. Artifact rejection: epochs with >100µV peak-to-peak are discarded.
  5. All v6 features retained: FBCSP 5-band, Platt calibration, 10 subjects.

Training vs Inference symmetry:
  Both classes extracted using the SAME method:
    raw.get_data() → pick motor channels → filter(0.5,50) → chunk into 1s windows
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

from config import MOTOR_CHANNEL_NAMES

MODEL_OUT = MODELS_DIR / "moabb_csp_lda.pkl"
STATS_OUT = DATA_DIR   / "moabb_stats.pkl"

# ── Constants ────────────────────────────────────────────────────
SFREQ = 160           # PhysioNet sample rate
EPOCH_SAMPLES = 160   # 1 second at 160Hz (matches simulator chunk_size)
N_SUBJECTS = 10       # 10 subjects for robust covariance estimates
ARTIFACT_THRESHOLD = 100e-6  # ±100µV peak-to-peak artifact rejection

# ── Motor-cortex channel selection (22 of 64) ───────────────────
# Sensorimotor strip: FC5–FC6, C5–C6, CP5–CP6 (covers ERD/ERS)
# Imported from config.py as single source of truth (MOTOR_CHANNEL_NAMES)

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


def extract_epochs_from_edf(edf_path, epoch_samples, event_aligned=False,
                            pick_channels=None, artifact_threshold=None):
    """
    Extract epochs from an EDF file using the SAME method as the simulator.
    
    If event_aligned=True: extract 1s windows starting at each motor event.
    If event_aligned=False: chunk the entire recording into 1s windows (for rest).
    
    pick_channels: list of channel names to select (None = all).
    artifact_threshold: reject epochs with peak-to-peak > this value (in Volts).
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Channel selection: pick only motor-cortex channels
    if pick_channels is not None:
        available = [ch for ch in pick_channels if ch in raw.ch_names]
        if len(available) > 0:
            raw.pick_channels(available)
    
    raw.filter(0.5, 50, verbose=False)  # Wider bandpass: preserves delta floor + gamma ceiling
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels = data.shape[0]
    
    epochs = []
    n_rejected = 0
    
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
            epoch = data[:, start:start + epoch_samples]
            # Artifact rejection: skip epochs with extreme peak-to-peak
            if artifact_threshold is not None:
                ptp = np.ptp(epoch, axis=1)  # peak-to-peak per channel
                bad_ratio = np.mean(ptp > artifact_threshold)
                if bad_ratio > 0.25:  # >25% channels are bad
                    n_rejected += 1
                    continue
            epochs.append(epoch)
    else:
        # Chunk entire recording into fixed windows (for baseline/rest)
        n_samples = data.shape[1]
        for start in range(0, n_samples - epoch_samples, epoch_samples):
            epoch = data[:, start:start + epoch_samples]
            if artifact_threshold is not None:
                ptp = np.ptp(epoch, axis=1)
                bad_ratio = np.mean(ptp > artifact_threshold)
                if bad_ratio > 0.25:
                    n_rejected += 1
                    continue
            epochs.append(epoch)
    
    raw.close()
    if n_rejected > 0:
        print(f"      (rejected {n_rejected} artifact epochs)", flush=True)
    return epochs, n_channels


def augment_epochs(epochs, labels=None, max_jitter=25, rng=None):
    """
    Time-jitter augmentation: create shifted copies of each epoch.
    Adds ±jitter versions, effectively tripling the dataset.
    If labels is provided, returns (augmented_epochs, augmented_labels).
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    augmented = list(epochs)  # Keep originals
    aug_labels = list(labels) if labels is not None else None
    
    n_times = epochs[0].shape[1] if len(epochs) > 0 else 0
    
    for i, epoch in enumerate(epochs):
        n_ch, n_t = epoch.shape
        label = labels[i] if labels is not None else None
        
        # Forward jitter
        jitter = rng.randint(10, max_jitter + 1)
        if jitter < n_t:
            shifted = np.zeros_like(epoch)
            shifted[:, :n_t - jitter] = epoch[:, jitter:]
            shifted[:, n_t - jitter:] = epoch[:, -1:]  # Pad with last value
            augmented.append(shifted)
            if aug_labels is not None:
                aug_labels.append(label)
        
        # Backward jitter
        jitter = rng.randint(10, max_jitter + 1)
        if jitter < n_t:
            shifted = np.zeros_like(epoch)
            shifted[:, jitter:] = epoch[:, :n_t - jitter]
            shifted[:, :jitter] = epoch[:, :1]  # Pad with first value
            augmented.append(shifted)
            if aug_labels is not None:
                aug_labels.append(label)
    
    if aug_labels is not None:
        return augmented, np.array(aug_labels)
    return augmented


def main():
    print("=" * 60, flush=True)
    print("  ORBIT AI -- Training Pipeline (v7.0 — Leakage-Free FBCSP)", flush=True)
    print("  Channel selection + artifact rejection + honest CV", flush=True)
    print("=" * 60, flush=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_epochs_path = DATA_DIR / "moabb_epochs_v7.pkl"
    
    if cache_epochs_path.exists():
        print("\n[1-3/6] Loading pre-extracted & balanced epochs from local cache...", flush=True)
        with open(cache_epochs_path, "rb") as f:
            cache = pickle.load(f)
        X = cache['X']
        y = cache['y']
        subject_ids = cache['subject_ids']
        n_channels = cache['n_channels']
        print(f"   Total epochs loaded: {len(y)} ({len(X)//2} per class)", flush=True)
        print(f"   Shape: {X.shape} ({n_channels} motor-cortex channels)", flush=True)
        subjects = sorted(set(subject_ids))
    else:
        subjects = list(range(1, N_SUBJECTS + 1))
        motor_runs = [4, 6, 8, 10, 12, 14]   # All left/right fist motor imagery runs
        rest_runs  = [1, 2]                    # Eyes-open + eyes-closed baselines
        
        all_motor_epochs = []   # list of (epoch_array, subject_id)
        all_rest_epochs  = []
        n_channels = None
        
        # ── 1. Extract MOTOR IMAGERY epochs ──────────────────────────────
        print(f"\n[1/6] Downloading & extracting MOTOR IMAGERY epochs...", flush=True)
        print(f"       Subjects: {len(subjects)}, Runs: {motor_runs}", flush=True)
        print(f"       Channel selection: {len(MOTOR_CHANNEL_NAMES)} motor-cortex channels", flush=True)
        
        for subj in subjects:
            for run in motor_runs:
                path = download_edf(subj, run)
                if path is None:
                    continue
                try:
                    epochs, nc = extract_epochs_from_edf(
                        path, EPOCH_SAMPLES, event_aligned=True,
                        pick_channels=MOTOR_CHANNEL_NAMES,
                        artifact_threshold=ARTIFACT_THRESHOLD,
                    )
                    if n_channels is None:
                        n_channels = nc
                    for ep in epochs:
                        all_motor_epochs.append((ep, subj))
                except Exception as e:
                    print(f"   [WARN] S{subj:03d}R{run:02d}: {e}", flush=True)
        
        print(f"   Extracted {len(all_motor_epochs)} motor imagery epochs (post-artifact-rejection)", flush=True)
        
        # ── 2. Extract REST epochs ───────────────────────────────────────
        print(f"\n[2/6] Downloading & extracting REST epochs...", flush=True)
        print(f"       Subjects: {len(subjects)}, Runs: {rest_runs}", flush=True)
        
        for subj in subjects:
            for run in rest_runs:
                path = download_edf(subj, run)
                if path is None:
                    continue
                try:
                    epochs, _ = extract_epochs_from_edf(
                        path, EPOCH_SAMPLES, event_aligned=False,
                        pick_channels=MOTOR_CHANNEL_NAMES,
                        artifact_threshold=ARTIFACT_THRESHOLD,
                    )
                    for ep in epochs:
                        all_rest_epochs.append((ep, subj))
                except Exception as e:
                    print(f"   [WARN] S{subj:03d}R{run:02d}: {e}", flush=True)
        
        print(f"   Extracted {len(all_rest_epochs)} rest epochs (post-artifact-rejection)", flush=True)
        
        if len(all_motor_epochs) == 0 or len(all_rest_epochs) == 0:
            print("[ERROR] Not enough data. Aborting.", flush=True)
            return
        
        # ── 3. Balance (NO augmentation here — augmentation moves inside CV) ──
        print(f"\n[3/6] Balancing dataset (augmentation deferred to inside CV)...", flush=True)
        
        rng = np.random.RandomState(42)
        
        n_balanced = min(len(all_motor_epochs), len(all_rest_epochs))
        
        motor_idx = rng.choice(len(all_motor_epochs), n_balanced, replace=False)
        rest_idx  = rng.choice(len(all_rest_epochs), n_balanced, replace=False)
        
        selected_motor = [all_motor_epochs[i] for i in motor_idx]
        selected_rest  = [all_rest_epochs[i] for i in rest_idx]
        
        X_motor = np.array([ep for ep, _ in selected_motor])
        X_rest  = np.array([ep for ep, _ in selected_rest])
        motor_subjects = [s for _, s in selected_motor]
        rest_subjects  = [s for _, s in selected_rest]
        
        X = np.concatenate([X_rest, X_motor], axis=0)
        y = np.concatenate([np.zeros(n_balanced, dtype=int), np.ones(n_balanced, dtype=int)])
        subject_ids = np.array(rest_subjects + motor_subjects)
        
        shuffle_idx = rng.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        subject_ids = subject_ids[shuffle_idx]
        
        print(f"   Total epochs: {len(y)} ({n_balanced} per class, NO augmentation leakage)", flush=True)
        print(f"   Shape: {X.shape} ({n_channels} motor-cortex channels)", flush=True)
        
        print("\n   Caching balanced epochs to disk...", flush=True)
        with open(cache_epochs_path, "wb") as f:
            pickle.dump({
                'X': X, 'y': y,
                'subject_ids': subject_ids,
                'n_channels': n_channels,
            }, f, protocol=4)
        print("   OK saved to moabb_epochs_v7.pkl", flush=True)
        
    # ── 4. Architecture tournament (leakage-free: augment INSIDE CV) ─
    print(f"\n[4/6] Architecture tournament (5-Fold CV, augmentation INSIDE folds)...", flush=True)
    print(f"       This gives HONEST accuracy — no augmented copies leak into test folds.", flush=True)
    
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
            print(f"   Evaluating {name:12} across 5 folds (augment train only)...", flush=True)
            fold_scores = []
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                X_train, y_train = X[train_idx], y[train_idx]
                X_test,  y_test  = X[test_idx],  y[test_idx]
                
                # Augment ONLY the training fold — test fold stays clean
                X_train_aug_list, y_train_aug = augment_epochs(list(X_train), labels=y_train, max_jitter=25,
                                             rng=np.random.RandomState(42 + fold_i))
                X_train_aug = np.array(X_train_aug_list)
                
                pipe_clone = Pipeline([(n, type(s)(**s.get_params())) for n, s in pipe.steps])
                pipe_clone.fit(X_train_aug, y_train_aug)
                fold_acc = pipe_clone.score(X_test, y_test)
                fold_scores.append(fold_acc)
                print(f"      Fold {fold_i+1}: {fold_acc*100:.2f}%", flush=True)
            
            scores = np.array(fold_scores)
            mean_acc = scores.mean()
            print(f"   - {name:12}: {mean_acc*100:6.2f}% (+/- {scores.std()*100:.2f}%) [HONEST]", flush=True)
            if mean_acc > best_score:
                best_score = mean_acc
                best_name = name
                best_pipe = pipe
                best_scores = scores
        except Exception as e:
            import traceback
            print(f"   - {name:12}: FAILED ({e})", flush=True)
            traceback.print_exc()

    print(f"\n   [WINNER] {best_name} ({best_score*100:.2f}%) — leakage-free", flush=True)
    pipeline = best_pipe

    # ── 5. Train final model on ALL data (with augmentation) + Platt ─
    print("\n[5/6] Fitting final model on full dataset with Platt calibration...", flush=True)
    
    # Split out 15% calibration set to prevent Platt scaling data leakage
    from sklearn.model_selection import train_test_split
    X_train_final, X_cal, y_train_final, y_cal = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    
    # Augment full training set for final base model
    X_aug_list, y_aug = augment_epochs(list(X_train_final), labels=y_train_final, max_jitter=25, rng=np.random.RandomState(42))
    X_aug = np.array(X_aug_list)
    
    # Fit base pipeline on augmented data
    pipeline.fit(X_aug, y_aug)
    
    # Wrap with Platt scaling using prefit and calibration set
    calibrated = CalibratedClassifierCV(pipeline, method='sigmoid', cv='prefit')
    calibrated.fit(X_cal, y_cal)

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
        'motor_channels'   : MOTOR_CHANNEL_NAMES,
        'uses_tangent_space': False,
        'classes'          : {0: 'IDLE', 1: 'FORWARD'},
        'cv_mean'          : float(best_scores.mean()),
        'cv_std'           : float(best_scores.std()),
        'cv_leakage_free'  : True,
        'artifact_rejection': True,
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
    print(f"  CV Accuracy  : {best_scores.mean()*100:.2f}% (LEAKAGE-FREE)", flush=True)
    print(f"  Subjects     : {len(subjects)}", flush=True)
    print(f"  Channels     : {n_channels} (motor-cortex selection)", flush=True)
    print(f"  Epoch length : {EPOCH_SAMPLES} samples (1s @ {SFREQ}Hz)", flush=True)
    print(f"  Pipeline     : {best_name} (Platt-calibrated)", flush=True)
    print(f"  Augmentation : 3× inside CV only (no leakage)", flush=True)
    print(f"  Artifact rej.: ±{ARTIFACT_THRESHOLD*1e6:.0f}µV peak-to-peak", flush=True)
    print(f"  Filter bands : {len(FILTER_BANDS)} sub-bands" if uses_fbcsp else "  Filter bands : Single broadband", flush=True)
    print(f"  Classes      : 0=IDLE, 1=FORWARD", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()
