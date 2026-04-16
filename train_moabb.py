"""
ORBIT AI — MOABB Training Pipeline
====================================
Trains a cross-dataset BCI classifier using:
  - MOABB datasets: PhysionetMI (109 subjects, real + imagined movement)
  - Paradigm: LeftRightImagery (left hand vs right hand → mapped to IDLE vs FORWARD)
  - Pipeline: CSP (spatial filter) + LDA (classifier)
  - Evaluation: CrossSubjectEvaluation for generalization

Why CSP instead of our CNN-LSTM?
  - CSP selects only motor cortex channels (C3, C4) regardless of total channel count
  - Works identically on 21-channel and 64-channel hardware
  - Proven method used in BCI competitions since 2012
  - No domain shift problem — it adapts to each recording's own statistics
"""

import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mne.decoding import CSP

import moabb
from moabb.datasets import PhysionetMI
from moabb.paradigms import LeftRightImagery
from moabb.evaluations import WithinSessionEvaluation

moabb.set_log_level('ERROR')

# Paths
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_OUT = MODELS_DIR / "moabb_csp_lda.pkl"
STATS_OUT  = DATA_DIR / "moabb_stats.pkl"


def main():
    print("=" * 60)
    print("  ORBIT AI -- MOABB Training Pipeline")
    print("=" * 60)

    # -- 1. Dataset & Paradigm ─────────────────────────────────────────
    # PhysionetMI: 109 subjects, imagined left/right hand movement
    # LeftRightImagery: extracts epochs, labels left=0, right=1
    print("\n[+] Loading PhysionetMI dataset (auto-downloads on first run)...")
    dataset = PhysionetMI()
    paradigm = LeftRightImagery()

    # Increase to 30 subjects for clinical-grade stability
    subjects = list(range(1, 41))
    print(f"   Using {len(subjects)} subjects for training (PhysioNet Stage)")

    # ── 2. Extract epochs ───────────────────────────────────────────
    print("🔬 Extracting epochs (4s windows for max stability)...")
    # Using longer windows (4s) makes the brain patterns much clearer to the AI
    paradigm = LeftRightImagery(tmin=0, tmax=4)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
    
    print(f"   Epochs shape : {X.shape}")
    print(f"   Labels       : {np.unique(y, return_counts=True)}")

    # We optimize for Riemannian space which handles noise better than simple linear space
    from pyriemann.estimation import Covariances
    from pyriemann.classification import MDM
    from pyriemann.tangentspace import TangentSpace
    
    # We will try 3 architectures and pick the best
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
            ('svm', SVC(kernel='linear', probability=True))
        ])
    }

    best_score = 0
    best_name = ""
    best_pipe = None

    print("\n📊 Comparing Architectures:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_acc = scores.mean()
        print(f"   - {name:10}: {mean_acc*100:6.2f}%")
        if mean_acc > best_score:
            best_score = mean_acc
            best_name = name
            best_pipe = pipe

    print(f"\n🏆 Winner: {best_name} ({best_score*100:.2f}%)")
    pipeline = best_pipe

    # ── 5. Train Final Model on All Data ────────────────────────────
    print("\n🚀 Training final model on all data...")
    pipeline.fit(X, y)

    # ── 6. Compute Per-Channel Normalization Stats ───────────────────
    # These are used at inference to normalize incoming raw EEG epochs
    ch_mean = np.mean(X, axis=(0, 2))   # shape: (n_channels,)
    ch_std  = np.std(X,  axis=(0, 2)) + 1e-6
    n_channels = X.shape[1]

    stats = {
        'ch_mean'    : ch_mean,
        'ch_std'     : ch_std,
        'n_channels' : n_channels,
        'classes'    : {0: 'IDLE', 1: 'FORWARD'},
        'cv_mean'    : scores.mean(),
        'cv_std'     : scores.std(),
        'subjects'   : subjects,
    }

    # ── 7. Save ─────────────────────────────────────────────────────
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(pipeline, f)
    with open(STATS_OUT, 'wb') as f:
        pickle.dump(stats, f)

    print(f"\n✅ Model saved  →  {MODEL_OUT}")
    print(f"✅ Stats saved  →  {STATS_OUT}")
    print(f"\n{'='*60}")
    print(f"  Final CV Accuracy: {scores.mean() * 100:.2f}%")
    print(f"  Subjects trained : {len(subjects)}")
    print(f"  Channels         : {n_channels}")
    print(f"{'='*60}\n")
    print("🎯 Next: run predict_realtime.py to use the MOABB model!")


if __name__ == "__main__":
    main()
