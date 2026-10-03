"""
ORBIT AI -- MOABB Training Pipeline (v7.0 -- Leakage-Free FBCSP)
==================================================================
CORE IMPROVEMENTS over v6.0:
  1. FIX: Augmentation moved INSIDE CV loop to eliminate data leakage.
     The v6 pipeline augmented before splitting, allowing augmented copies
     of the same epoch into both train and test folds (inflating accuracy).
  2. Channel Selection: 64 -> 22 motor-cortex channels (sensorimotor strip).
     Reduces model size from ~3.5 GB to ~200 MB and removes noise channels.
  3. Subject labels preserved: enables Leave-One-Subject-Out (LOSO) CV.
  4. Artifact rejection: epochs with >100uV peak-to-peak are discarded.
  5. All v6 features retained: FBCSP 5-band, Platt calibration, 10 subjects.

Training vs Inference symmetry:
  Both classes extracted using the SAME method:
    raw.get_data() -> pick motor channels -> filter(0.5,50) -> chunk into 1s windows
  FORWARD class: chunks from event-marked motor imagery periods in R04/R06/R08/R10/R12/R14
  IDLE class:    chunks from continuous baseline recordings R01/R02
"""

import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

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
SFREQ = 250           # ADS1299 hardware sample rate (PhysioNet 160Hz resampled up)
EPOCH_SAMPLES = 250   # 1 second at 250Hz (matches ADS1299 output)
N_SUBJECTS = 10       # 10 subjects for robust covariance estimates
ARTIFACT_THRESHOLD = 100e-6  # ±100uV peak-to-peak artifact rejection

# ── Motor-cortex channel selection (22 of 64) ───────────────────
# Sensorimotor strip: FC5-FC6, C5-C6, CP5-CP6 (covers ERD/ERS)
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
# FBCSP Transformer -- filters each epoch into sub-bands and
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
                            pick_channels=None, artifact_threshold=None,
                            return_event_types=False):
    """
    Extract epochs from an EDF file using the SAME method as the simulator.
    
    If event_aligned=True: extract 1s windows starting at each motor event.
    If event_aligned=False: chunk the entire recording into 1s windows (for rest).
    
    pick_channels: list of channel names to select (None = all).
    artifact_threshold: reject epochs with peak-to-peak > this value (in Volts).
    return_event_types: if True, returns (epochs, n_channels, event_types) where
                        event_types[i] is 'T1' or 'T2' for each epoch.
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    
    # Channel selection: pick only motor-cortex channels
    if pick_channels is not None:
        available = [ch for ch in pick_channels if ch in raw.ch_names]
        if len(available) > 0:
            raw.pick_channels(available)
    
    raw.filter(0.5, 50, verbose=False)  # Wider bandpass: preserves delta floor + gamma ceiling
    # Resample to match ADS1299 hardware rate (PhysioNet 160Hz → 250Hz)
    if abs(raw.info['sfreq'] - SFREQ) > 1.0:
        raw.resample(SFREQ, verbose=False)
    data = raw.get_data()  # (n_channels, n_samples)
    n_channels = data.shape[0]
    
    epochs = []
    event_types = []  # 'T1' or 'T2' per epoch
    n_rejected = 0
    
    if event_aligned:
        # Extract windows aligned to motor imagery events (T1=left, T2=right)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Build reverse map: event_int_id -> annotation name (e.g., 'T1', 'T2')
        id_to_name = {eid: name for name, eid in event_id.items()}
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
            # Determine event type from annotation name
            ann_name = id_to_name.get(ev[2], '')
            if 'T1' in ann_name:
                event_types.append('T1')
            elif 'T2' in ann_name:
                event_types.append('T2')
            else:
                event_types.append('UNKNOWN')
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
            event_types.append('REST')
    
    raw.close()
    if n_rejected > 0:
        print(f"      (rejected {n_rejected} artifact epochs)", flush=True)
    if return_event_types:
        return epochs, n_channels, event_types
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
    print("  ORBIT AI -- Training Pipeline (v8.0 -- 4-Class Motor Imagery)", flush=True)
    print("  IDLE / LEFT / RIGHT / FORWARD -- Leakage-Free FBCSP", flush=True)
    print("=" * 60, flush=True)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    cache_epochs_path = DATA_DIR / "moabb_epochs_v9_4class_250hz.pkl"
    
    if cache_epochs_path.exists():
        print("\n[1-3/6] Loading pre-extracted & balanced 4-class epochs from local cache...", flush=True)
        with open(cache_epochs_path, "rb") as f:
            cache = pickle.load(f)
        X = cache['X']
        y = cache['y']
        subject_ids = cache['subject_ids']
        n_channels = cache['n_channels']
        for cls_id in sorted(set(y)):
            cls_count = np.sum(y == cls_id)
            cls_name = {0: 'IDLE', 1: 'LEFT', 2: 'RIGHT', 3: 'FORWARD'}[cls_id]
            print(f"   Class {cls_id} ({cls_name}): {cls_count} epochs", flush=True)
        print(f"   Total epochs: {len(y)}", flush=True)
        print(f"   Shape: {X.shape} ({n_channels} motor-cortex channels)", flush=True)
        subjects = sorted(set(subject_ids))
    else:
        subjects = list(range(1, N_SUBJECTS + 1))
        
        # ── PhysioNet EEGMMIDB Run Structure ──────────────────────────────
        # Left/right fist runs:  R03, R07, R11 (execution), R05, R09, R13 (imagery)
        #   T1 = left fist, T2 = right fist
        # Feet/fists runs:       R04, R08, R12 (execution), R06, R10, R14 (imagery)
        #   T1 = both fists (DISCARDED), T2 = both feet (FORWARD)
        # Rest runs:             R01 (eyes open), R02 (eyes closed)
        
        left_right_runs = [3, 7, 11, 5, 9, 13]   # T1=left, T2=right
        feet_runs = [4, 8, 12, 6, 10, 14]          # T2=feet (T1=fists discarded)
        rest_runs = [1, 2]                          # Continuous baseline
        
        all_left_epochs = []     # class 1: (epoch, subject_id)
        all_right_epochs = []    # class 2: (epoch, subject_id)
        all_forward_epochs = []  # class 3: (epoch, subject_id)
        all_rest_epochs = []     # class 0: (epoch, subject_id)
        n_channels = None
        n_discarded_fists = 0
        
        # ── 1. Extract LEFT + RIGHT epochs from fist runs ─────────────────
        print(f"\n[1/6] Downloading & extracting LEFT/RIGHT epochs (fist runs)...", flush=True)
        print(f"       Subjects: {len(subjects)}, Runs: {left_right_runs}", flush=True)
        print(f"       T1 events -> LEFT, T2 events -> RIGHT", flush=True)
        print(f"       Channel selection: {len(MOTOR_CHANNEL_NAMES)} motor-cortex channels", flush=True)
        
        for subj in subjects:
            for run in left_right_runs:
                path = download_edf(subj, run)
                if path is None:
                    continue
                try:
                    epochs, nc, event_types = extract_epochs_from_edf(
                        path, EPOCH_SAMPLES, event_aligned=True,
                        pick_channels=MOTOR_CHANNEL_NAMES,
                        artifact_threshold=ARTIFACT_THRESHOLD,
                        return_event_types=True,
                    )
                    if n_channels is None:
                        n_channels = nc
                    for ep, etype in zip(epochs, event_types):
                        if etype == 'T1':
                            all_left_epochs.append((ep, subj))
                        elif etype == 'T2':
                            all_right_epochs.append((ep, subj))
                except Exception as e:
                    print(f"   [WARN] S{subj:03d}R{run:02d}: {e}", flush=True)
        
        print(f"   LEFT epochs:  {len(all_left_epochs)}", flush=True)
        print(f"   RIGHT epochs: {len(all_right_epochs)}", flush=True)
        
        # ── 2. Extract FORWARD epochs from feet runs ──────────────────────
        print(f"\n[2/6] Downloading & extracting FORWARD epochs (feet runs)...", flush=True)
        print(f"       Subjects: {len(subjects)}, Runs: {feet_runs}", flush=True)
        print(f"       T2 events -> FORWARD, T1 events (both fists) -> DISCARDED", flush=True)
        
        for subj in subjects:
            for run in feet_runs:
                path = download_edf(subj, run)
                if path is None:
                    continue
                try:
                    epochs, nc, event_types = extract_epochs_from_edf(
                        path, EPOCH_SAMPLES, event_aligned=True,
                        pick_channels=MOTOR_CHANNEL_NAMES,
                        artifact_threshold=ARTIFACT_THRESHOLD,
                        return_event_types=True,
                    )
                    if n_channels is None:
                        n_channels = nc
                    for ep, etype in zip(epochs, event_types):
                        if etype == 'T2':
                            all_forward_epochs.append((ep, subj))
                        elif etype == 'T1':
                            n_discarded_fists += 1
                            # Discard "both fists" -- not a useful command
                except Exception as e:
                    print(f"   [WARN] S{subj:03d}R{run:02d}: {e}", flush=True)
        
        print(f"   FORWARD epochs: {len(all_forward_epochs)}", flush=True)
        print(f"   Discarded (both fists): {n_discarded_fists}", flush=True)
        
        # ── 3. Extract REST epochs ────────────────────────────────────────
        print(f"\n[3/6] Downloading & extracting REST/IDLE epochs...", flush=True)
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
        
        print(f"   IDLE/REST epochs: {len(all_rest_epochs)}", flush=True)
        
        # ── Validation ────────────────────────────────────────────────────
        class_counts = {
            'IDLE': len(all_rest_epochs),
            'LEFT': len(all_left_epochs),
            'RIGHT': len(all_right_epochs),
            'FORWARD': len(all_forward_epochs),
        }
        print(f"\n   Pre-balance counts: {class_counts}", flush=True)
        
        if any(c == 0 for c in class_counts.values()):
            print("[ERROR] One or more classes have zero epochs. Aborting.", flush=True)
            return
        
        # ── 4. Balance all 4 classes (NO augmentation -- deferred to CV) ──
        print(f"\n[4/6] Balancing 4-class dataset (augmentation deferred to inside CV)...", flush=True)
        
        rng = np.random.RandomState(42)
        n_balanced = min(class_counts.values())
        
        def sample_class(epoch_list, n):
            idx = rng.choice(len(epoch_list), n, replace=False)
            return [epoch_list[i] for i in idx]
        
        selected_rest = sample_class(all_rest_epochs, n_balanced)
        selected_left = sample_class(all_left_epochs, n_balanced)
        selected_right = sample_class(all_right_epochs, n_balanced)
        selected_forward = sample_class(all_forward_epochs, n_balanced)
        
        X_rest    = np.array([ep for ep, _ in selected_rest])
        X_left    = np.array([ep for ep, _ in selected_left])
        X_right   = np.array([ep for ep, _ in selected_right])
        X_forward = np.array([ep for ep, _ in selected_forward])
        
        rest_subjects    = [s for _, s in selected_rest]
        left_subjects    = [s for _, s in selected_left]
        right_subjects   = [s for _, s in selected_right]
        forward_subjects = [s for _, s in selected_forward]
        
        X = np.concatenate([X_rest, X_left, X_right, X_forward], axis=0)
        y = np.concatenate([
            np.full(n_balanced, 0, dtype=int),   # IDLE
            np.full(n_balanced, 1, dtype=int),   # LEFT
            np.full(n_balanced, 2, dtype=int),   # RIGHT
            np.full(n_balanced, 3, dtype=int),   # FORWARD
        ])
        subject_ids = np.array(rest_subjects + left_subjects + right_subjects + forward_subjects)
        
        shuffle_idx = rng.permutation(len(y))
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        subject_ids = subject_ids[shuffle_idx]
        
        print(f"   Total epochs: {len(y)} ({n_balanced} per class × 4 classes, NO augmentation leakage)", flush=True)
        print(f"   Shape: {X.shape} ({n_channels} motor-cortex channels)", flush=True)
        
        print("\n   Caching balanced 4-class epochs to disk...", flush=True)
        with open(cache_epochs_path, "wb") as f:
            pickle.dump({
                'X': X, 'y': y,
                'subject_ids': subject_ids,
                'n_channels': n_channels,
            }, f, protocol=4)
        print("   OK saved to moabb_epochs_v8_4class.pkl", flush=True)
        
    # ── 5. Architecture tournament (leakage-free: augment INSIDE CV) ─
    print(f"\n[5/6] Architecture tournament (5-Fold CV, augmentation INSIDE folds)...", flush=True)
    print(f"       4-class: IDLE / LEFT / RIGHT / FORWARD", flush=True)
    print(f"       This gives HONEST accuracy -- no augmented copies leak into test folds.", flush=True)
    
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
                
                # Augment ONLY the training fold -- test fold stays clean
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

    print(f"\n   [WINNER] {best_name} ({best_score*100:.2f}%) -- leakage-free, 4-class", flush=True)
    pipeline = best_pipe

    # ── 6. Train final model on ALL data (with augmentation) + Platt ─
    print("\n[6/6] Fitting final model on full dataset with Platt calibration...", flush=True)
    
    # Augment full dataset for final base model
    X_aug_list, y_aug = augment_epochs(list(X), labels=y, max_jitter=25, rng=np.random.RandomState(42))
    X_aug = np.array(X_aug_list)
    
    # Wrap with Platt scaling — sklearn handles calibration split internally via CV
    # (cv='prefit' removed in newer sklearn; cv=3 achieves the same with less leakage risk)
    calibrated = CalibratedClassifierCV(pipeline, method='sigmoid', cv=3)
    calibrated.fit(X_aug, y_aug)

    # ── 7. Save ──────────────────────────────────────────────────────
    ch_mean = np.mean(X, axis=(0, 2))
    ch_std  = np.std(X,  axis=(0, 2)) + 1e-6

    uses_fbcsp = "FBCSP" in best_name

    CLASS_MAP = {0: 'IDLE', 1: 'LEFT', 2: 'RIGHT', 3: 'FORWARD'}

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
        'classes'          : CLASS_MAP,
        'n_classes'        : 4,
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
    print(f"  CV Accuracy  : {best_scores.mean()*100:.2f}% (LEAKAGE-FREE, 4-CLASS)", flush=True)
    print(f"  Chance Level : 25.00%", flush=True)
    print(f"  Subjects     : {len(subjects)}", flush=True)
    print(f"  Channels     : {n_channels} (motor-cortex selection)", flush=True)
    print(f"  Epoch length : {EPOCH_SAMPLES} samples (1s @ {SFREQ}Hz)", flush=True)
    print(f"  Pipeline     : {best_name} (Platt-calibrated)", flush=True)
    print(f"  Augmentation : 3× inside CV only (no leakage)", flush=True)
    print(f"  Artifact rej.: ±{ARTIFACT_THRESHOLD*1e6:.0f}uV peak-to-peak", flush=True)
    print(f"  Filter bands : {len(FILTER_BANDS)} sub-bands" if uses_fbcsp else "  Filter bands : Single broadband", flush=True)
    print(f"  Classes      : 0=IDLE, 1=LEFT, 2=RIGHT, 3=FORWARD", flush=True)
    print(f"{'='*60}\n", flush=True)


if __name__ == "__main__":
    main()

