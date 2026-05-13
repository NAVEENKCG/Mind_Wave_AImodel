"""
ORBIT AI — MOABB Training Pipeline (v5.0 — Definitive Expert Fix)
==================================================================
CORE PRINCIPLE: Training data must be processed IDENTICALLY to inference data.

The previous versions failed because:
  v1-v3: Trained on left_hand vs right_hand (no IDLE class)
  v4.0:  Used MOABB paradigm for motor epochs (which applies internal
         preprocessing like event alignment + baseline correction) but 
         raw download for rest epochs. The resulting covariance matrices
         have different statistical structure → 100% CV but 0% generalization.

THIS FIX: 
  Both classes are extracted using the EXACT SAME method:
    raw.get_data() → filter(1,45) → chunk into 1s windows
  
  FORWARD class: chunks from event-marked motor imagery periods in R04/R08/R12
  IDLE class:    chunks from continuous baseline recording R01
  
  No MOABB paradigm used for data extraction. Only for dataset download.
"""

import numpy as np
import pickle
import warnings
import urllib.request
import tempfile
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC

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
N_SUBJECTS = 5     # 5 subjects × 4 runs = enough for demo, fast to download


def download_edf(subject_id, run_id):
    """Download a specific PhysioNet EEG Motor Imagery run."""
    url = f"https://physionet.org/files/eegmmidb/1.0.0/S{subject_id:03d}/S{subject_id:03d}R{run_id:02d}.edf"
    print(f"      Downloading S{subject_id:03d}R{run_id:02d}...", end="", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
            with urllib.request.urlopen(req, timeout=60) as resp:
                tmp.write(resp.read())
            print(" OK", flush=True)
            return tmp.name
    except Exception as e:
        print(f" FAILED ({e})", flush=True)
        return None


def extract_epochs_from_edf(edf_path, epoch_samples, event_aligned=False):
    """
    Extract epochs from an EDF file using the SAME method as the simulator.
    
    If event_aligned=True: extract 1s windows starting at each motor event.
    If event_aligned=False: chunk the entire recording into 1s windows (for rest).
    """
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(1, 45, verbose=False)
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


def main():
    print("=" * 60)
    print("  ORBIT AI -- Training Pipeline (v5.0)")
    print("  Identical preprocessing for train & inference")
    print("=" * 60)
    
    subjects = list(range(1, N_SUBJECTS + 1))
    motor_runs = [4, 8, 12]  # Left/right fist motor imagery runs
    rest_runs  = [1]          # Eyes-open baseline
    
    all_motor_epochs = []
    all_rest_epochs  = []
    n_channels = None
    
    # ── 1. Extract MOTOR IMAGERY epochs ──────────────────────────────
    print(f"\n[1/4] Downloading & extracting MOTOR IMAGERY epochs...")
    print(f"       Subjects: {len(subjects)}, Runs: {motor_runs}")
    
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
                print(f"   [WARN] S{subj:03d}R{run:02d}: {e}")
            finally:
                try: Path(path).unlink(missing_ok=True)
                except: pass
    
    print(f"   Extracted {len(all_motor_epochs)} motor imagery epochs")
    
    # ── 2. Extract REST epochs ───────────────────────────────────────
    print(f"\n[2/4] Downloading & extracting REST epochs...")
    print(f"       Subjects: {len(subjects)}, Runs: {rest_runs}")
    
    for subj in subjects:
        for run in rest_runs:
            path = download_edf(subj, run)
            if path is None:
                continue
            try:
                epochs, _ = extract_epochs_from_edf(path, EPOCH_SAMPLES, event_aligned=False)
                all_rest_epochs.extend(epochs)
            except Exception as e:
                print(f"   [WARN] S{subj:03d}R{run:02d}: {e}")
            finally:
                try: Path(path).unlink(missing_ok=True)
                except: pass
    
    print(f"   Extracted {len(all_rest_epochs)} rest epochs")
    
    if len(all_motor_epochs) == 0 or len(all_rest_epochs) == 0:
        print("[ERROR] Not enough data. Aborting.")
        return
    
    # ── 3. Balance and combine ───────────────────────────────────────
    print(f"\n[3/4] Balancing dataset...")
    
    X_motor = np.array(all_motor_epochs)
    X_rest  = np.array(all_rest_epochs)
    
    n_balanced = min(len(X_motor), len(X_rest))
    rng = np.random.RandomState(42)
    
    X_motor = X_motor[rng.choice(len(X_motor), n_balanced, replace=False)]
    X_rest  = X_rest[rng.choice(len(X_rest), n_balanced, replace=False)]
    
    X = np.concatenate([X_rest, X_motor], axis=0)
    y = np.concatenate([np.zeros(n_balanced, dtype=int), np.ones(n_balanced, dtype=int)])
    
    shuffle_idx = rng.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    print(f"   Total epochs: {len(y)} ({n_balanced} per class)")
    print(f"   Shape: {X.shape}")
    
    # ── 4. Architecture tournament ───────────────────────────────────
    print(f"\n[4/4] Architecture tournament (5-Fold CV)...")
    
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
        "TS_SVM": Pipeline([
            ('cov', Covariances(estimator='lwf')),
            ('ts', TangentSpace(metric='riemann')),
            ('svm', SVC(kernel='rbf', probability=True, C=1.0, gamma='scale'))
        ]),
    }
    
    best_score = 0
    best_name = ""
    best_pipe = None
    best_scores = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = scores.mean()
        print(f"   - {name:10}: {mean_acc*100:6.2f}% (+/- {scores.std()*100:.2f}%)")
        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name
            best_pipe = pipe
            best_scores = scores

    print(f"\n   [WINNER] {best_name} ({best_score*100:.2f}%)")
    pipeline = best_pipe

    # ── 5. Train final model ─────────────────────────────────────────
    print("\n[TRAIN] Fitting final model on all data...")
    pipeline.fit(X, y)

    # ── 6. Save ──────────────────────────────────────────────────────
    ch_mean = np.mean(X, axis=(0, 2))
    ch_std  = np.std(X,  axis=(0, 2)) + 1e-6

    # Riemannian pipelines (Cov → TS → classifier) must NOT have external
    # z-score normalization. The Covariances estimator handles scale internally.
    uses_tangent_space = False

    stats = {
        'ch_mean'          : ch_mean,
        'ch_std'           : ch_std,
        'n_channels'       : n_channels,
        'n_times'          : EPOCH_SAMPLES,
        'sfreq'            : float(SFREQ),
        'uses_tangent_space': uses_tangent_space,
        'classes'          : {0: 'IDLE', 1: 'FORWARD'},
        'cv_mean'          : float(best_scores.mean()),
        'cv_std'           : float(best_scores.std()),
        'subjects'         : subjects,
        'pipeline'         : best_name,
    }

    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(STATS_OUT, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\n[SAVED] Model  -> {MODEL_OUT}")
    print(f"[SAVED] Stats  -> {STATS_OUT}")
    print(f"\n{'='*60}")
    print(f"  CV Accuracy  : {best_scores.mean()*100:.2f}%")
    print(f"  Subjects     : {len(subjects)}")
    print(f"  Channels     : {n_channels}")
    print(f"  Epoch length : {EPOCH_SAMPLES} samples (1s @ {SFREQ}Hz)")
    print(f"  Pipeline     : {best_name}")
    print(f"  Classes      : 0=IDLE, 1=FORWARD")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
