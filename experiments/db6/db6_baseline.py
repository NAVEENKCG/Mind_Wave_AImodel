"""
db6_baseline.py (v2) - plain EMG baseline on NinaPro DB6, no drift handling yet.

This is the number the drift work has to beat. Two measurements:
  1. WITHIN-SESSION: train on some repetitions, test on the others, SAME session.
     The ceiling: how well plain EMG classification does when nothing has drifted.
  2. ACCURACY-VS-TIME-GAP: train once on the FIRST session (D1 T1), then test on
     every other session using ALL its repetitions (the model has never seen them,
     so nothing leaks). The first session is tested on its held-out reps only.

Changes from v1 (found by testing v1 on DB6-shaped data):
  - Rest windows are kept. If rerepetition is 0 during rest, v1 silently dropped
    every rest window (no rest class, rest false activations = nan). v2 gives each
    rest sample the index of the repetition it follows, and aborts if a split has
    no rest windows.
  - ZC and SSC use the textbook definitions. v1's "zc" counted slope sign changes
    and its "ssc" counted monotonic slope pairs (the opposite), so they were
    near-duplicates and true zero crossings were never computed.
  - Files are sorted by (day, session) and the training session is checked to be
    D1 T1, not "whichever file sorts first".
  - Train/test objects are printed per session (DB6 splits repetitions across two
    objects; a split that puts one object only in test is a confound).
  - Features are computed per session and the raw windows are freed (v1 kept every
    session's windows in memory, about 1.5 GB for one subject).
  - Rest-only RMS, active-only RMS and their ratio (SNR) per channel, so dead or
    weak channels show up in numbers, not only in plots.

Pipeline: causal 20-250 Hz bandpass -> downsample 2000 -> 1000 Hz -> 200 ms windows,
100 ms step (windows crossing a label change are dropped) -> MAV / WL / ZC / SSC
per channel -> LDA.

Install:   pip install numpy scipy scikit-learn matplotlib

Usage (repo root, data from db6_explore.py in data/db6_subject):
  python experiments/db6/db6_baseline.py --dir data/db6_subject
  python experiments/db6/db6_baseline.py --dir data/db6_subject --drop-channels 7
  python experiments/db6/db6_baseline.py --dir data/db6_subject --train-reps 1-6 --test-reps 7-12

Choose --train-reps/--test-reps AFTER reading the object lines in the output.
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, sosfilt

FS_RAW = 2000
FS_OUT = 1000
BAND = (20.0, 250.0)
WIN_MS = 200
STEP_MS = 100
NAME_RE = re.compile(r"S(\d+)_D(\d+)_T(\d+)", re.IGNORECASE)


# ----------------------------------------------------------------- loading
def parse_reps(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def load_mat(path):
    try:
        raw = sio.loadmat(str(path))
        return {k: np.asarray(v) for k, v in raw.items() if not k.startswith("__")}
    except NotImplementedError:
        try:
            import h5py
        except ImportError:
            raise SystemExit("This .mat is v7.3 - run: pip install h5py")
        out = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    out[k] = np.array(f[k]).T
        return out


def col(d, *names):
    for n in names:
        if n in d:
            return np.asarray(d[n]).reshape(-1)
    return None


def real_channels(emg, tol=1e-9):
    std = emg.std(axis=0)
    return [i for i, s in enumerate(std) if s > tol]


def ffill_nonzero(v):
    """Give every zero sample the last non-zero value before it (leading zeros get the
    first non-zero value). Used so rest samples inherit the repetition/object they follow."""
    v = np.asarray(v).astype(int)
    nz = np.flatnonzero(v > 0)
    if nz.size == 0:
        return v
    idx = np.where(v > 0, np.arange(len(v)), 0)
    np.maximum.accumulate(idx, out=idx)
    out = v[idx]
    out[: nz[0]] = v[nz[0]]
    return out


# ------------------------------------------------------------ signal chain
def bandpass_causal(emg, fs):
    sos = butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, emg, axis=0)


def make_windows(sig, lab, rep, obj, fs):
    """200 ms windows, 100 ms step; keep metadata-constant windows.
    Returns X (n, win, ch), y, repetition per window, object per window (modes)."""
    win = int(WIN_MS * fs / 1000)
    step = int(STEP_MS * fs / 1000)
    keep, y, r, o = [], [], [], []
    for s in range(0, len(sig) - win + 1, step):
        w = lab[s:s + win]
        rw = rep[s:s + win]
        ow = obj[s:s + win]
        if np.all(w == w[0]) and np.all(rw == rw[0]) and np.all(ow == ow[0]):
            keep.append(s)
            y.append(w[0])
            r.append(rw[0])
            o.append(ow[0])
    if not keep:
        return np.empty((0, win, sig.shape[1])), np.array([]), np.array([]), np.array([])
    X = np.stack([sig[s:s + win] for s in keep])
    return X, np.array(y).astype(int), np.array(r), np.array(o)


def zc_threshold(X):
    """Per-channel threshold, 2% of that channel's std on TRAIN windows."""
    return 0.02 * X.std(axis=(0, 1))


def _features_block(X, thr):
    mav = np.abs(X).mean(axis=1)
    dx = np.diff(X, axis=1)
    wl = np.abs(dx).sum(axis=1)
    zc = ((X[:, :-1] * X[:, 1:] < 0) & (np.abs(dx) >= thr)).sum(axis=1)       # zero crossings of the signal
    a = X[:, 1:-1] - X[:, :-2]
    b = X[:, 1:-1] - X[:, 2:]
    ssc = ((a * b > 0) & ((np.abs(a) >= thr) | (np.abs(b) >= thr))).sum(axis=1)  # slope sign changes
    return np.hstack([mav, wl, zc, ssc])


def features(X, thr, chunk=1000):
    """MAV, WL, ZC, SSC per channel -> (n_windows, 4 * n_channels). Chunked to bound memory."""
    if len(X) == 0:
        return np.empty((0, 4 * X.shape[2]))
    return np.vstack([_features_block(X[i:i + chunk], thr) for i in range(0, len(X), chunk)])


# ----------------------------------------------------------------- metrics
def report(tag, y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    acc = float((y_true == y_pred).mean())
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    rest = y_true == 0
    fa = float((y_pred[rest] != 0).mean()) if rest.any() else float("nan")
    print(f"    {tag:34s} acc={acc:.3f}  balanced={bacc:.3f}  rest-false-act={fa:.3f}")
    return bacc, fa


def train_lda(F, y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(F, y)
    return clf


def check_split(name, y, tr, te):
    """Abort loudly if rest is missing from train or test; warn on missing grasp classes."""
    for tag, m in (("train", tr), ("test", te)):
        classes, counts = np.unique(y[m], return_counts=True)
        desc = ", ".join(f"{int(c)}:{n}" for c, n in zip(classes, counts))
        print(f"    {tag} windows by label  {desc}")
        if 0 not in classes:
            raise SystemExit(
                f"\nERROR: {name}: no REST windows (label 0) in the {tag} split. The rest class is "
                "required for rest-false-activation numbers. Check how rerepetition is coded at rest "
                "(print np.unique(rerepetition) and the values during label 0).")
    miss = set(np.unique(y[tr])) ^ set(np.unique(y[te]))
    if miss:
        print(f"    WARNING: labels {sorted(int(m) for m in miss)} appear in only one of train/test.")


def object_report(o, tr, te):
    ot = sorted(set(int(v) for v in np.unique(o[tr])) - {0})
    oe = sorted(set(int(v) for v in np.unique(o[te])) - {0})
    print(f"    objects in train reps: {ot} | in test reps: {oe}")
    if set(ot) != set(oe):
        print("    WARNING: train and test use different objects. Object is confounded with the "
              "split - choose --train-reps/--test-reps that include every object in both.")


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/db6_subject", help="folder with extracted .mat files")
    ap.add_argument("--out", default=None, help="output folder (default: <dir>/baseline)")
    ap.add_argument("--fs", type=float, default=FS_RAW)
    ap.add_argument("--drop-channels", nargs="*", type=int, default=[],
                    help="0-based channels to exclude, e.g. --drop-channels 7")
    ap.add_argument("--train-reps", default="1-8", help="e.g. 1-8 or 1,2,3,4 (default 1-8)")
    ap.add_argument("--test-reps", default="9-12", help="e.g. 9-12 (default 9-12)")
    ap.add_argument("--cross-test", choices=["all", "test-reps"], default="all",
                    help="reps of OTHER sessions used for the time-gap curve (default: all 12)")
    a = ap.parse_args()
    train_reps, test_reps = parse_reps(a.train_reps), parse_reps(a.test_reps)
    if set(train_reps) & set(test_reps):
        raise SystemExit("train and test repetitions overlap")

    t0 = time.time()
    root = Path(a.dir)
    out = Path(a.out) if a.out else root / "baseline"
    out.mkdir(parents=True, exist_ok=True)

    def key(f):
        m = NAME_RE.search(f.name)
        return (int(m.group(2)), int(m.group(3)), str(f)) if m else (99, 99, str(f))

    files = sorted(root.rglob("*.mat"), key=key)
    files = [f for f in files if f.parent.name != "baseline"]
    if not files:
        raise SystemExit(f"No .mat files under {root}.")
    print(f"Found {len(files)} .mat files under {root}  (sorted by day, session)")
    print(f"Train reps {train_reps} | test reps {test_reps}")

    sessions, rms_rows = [], []
    channels, thr0 = None, None
    for f in files:
        m = NAME_RE.search(f.name)
        day = int(m.group(2)) if m else 0
        sess = int(m.group(3)) if m else 0
        d = load_mat(f)
        if "emg" not in d:
            print(f"  {f.name}: no 'emg' variable, skipped")
            continue
        lab = col(d, "restimulus", "stimulus")
        rep = col(d, "rerepetition", "repetition")
        obj = col(d, "reobject", "object")
        if lab is None or rep is None or obj is None:
            print(f"  {f.name}: missing label variables, skipped")
            continue
        emg = d["emg"]
        if channels is None:
            channels = [c for c in real_channels(emg) if c not in a.drop_channels]
            print(f"Using {len(channels)} channels: {channels}"
                  + (f"  (dropped {a.drop_channels})" if a.drop_channels else ""))
        sig = bandpass_causal(emg[:, channels], a.fs)
        del emg, d
        lab = lab.astype(int)
        rep = ffill_nonzero(rep)
        obj = ffill_nonzero(obj)
        if a.fs != FS_OUT:
            q = int(round(a.fs / FS_OUT))
            if abs(a.fs / q - FS_OUT) > 1e-6:
                raise SystemExit(f"Cannot downsample {a.fs} to {FS_OUT} Hz by an integer factor")
            sig, lab, rep, obj = sig[::q], lab[::q], rep[::q], obj[::q]

        rest, act = lab == 0, lab > 0
        r_rms = np.sqrt((sig[rest] ** 2).mean(axis=0)) if rest.any() else np.full(sig.shape[1], np.nan)
        a_rms = np.sqrt((sig[act] ** 2).mean(axis=0)) if act.any() else np.full(sig.shape[1], np.nan)
        rms_rows.append((f.name, r_rms, a_rms))

        X, y, r, o = make_windows(sig, lab, rep, obj, FS_OUT)
        del sig
        tr = np.isin(r, train_reps)
        te = np.isin(r, test_reps)
        print(f"\n  {f.name}: day={day} T{sess}  windows={len(y)}  (train-reps {int(tr.sum())}, test-reps {int(te.sum())})")
        check_split(f.name, y, tr, te)
        object_report(o, tr, te)

        thr_own = zc_threshold(X[tr])
        F_own = features(X, thr_own)
        if thr0 is None:                       # first (sorted) session = the training session
            thr0 = thr_own
            if (day, sess) != (1, 1):
                print(f"    WARNING: the first session found is D{day} T{sess}, not D1 T1. "
                      "The drift curve will be measured from this session.")
            F_thr0 = F_own
        else:
            F_thr0 = features(X, thr0)         # frozen train-session threshold, as a live device would have
        del X
        sessions.append(dict(name=f.name, day=day, sess=sess, y=y, tr=tr, te=te,
                             F_own=F_own, F_thr0=F_thr0))

    if not sessions:
        raise SystemExit("No usable sessions.")

    # ---- channel quality: rest noise floor, active level, SNR
    print("\nChannel quality per file (filtered signal).")
    print("  rest-RMS ratio = vs first file's rest RMS; SNR = active RMS / rest RMS (about 1 = no grasp information)")
    first_rest = rms_rows[0][1]
    print("  file           " + " ".join(f"ch{c:<5d}" for c in channels))
    for name, r_rms, a_rms in rms_rows:
        ratio = r_rms / np.maximum(first_rest, 1e-30)
        snr = a_rms / np.maximum(r_rms, 1e-30)
        print(f"  {name:14s} rest-ratio " + " ".join(f"{v:6.2f}" for v in ratio))
        print(f"  {'':14s} SNR        " + " ".join(f"{v:6.2f}" for v in snr))
        flags = [c for c, rr, ss in zip(channels, ratio, snr) if rr < 0.2 or rr > 5 or ss < 1.5]
        if flags:
            print(f"  {'':14s} look at channels: {flags}")

    # ---- within-session baseline (ceiling)
    print(f"\n1) WITHIN-SESSION baseline (train reps {a.train_reps} / test reps {a.test_reps}, same session):")
    within = {}
    for s in sessions:
        clf = train_lda(s["F_own"][s["tr"]], s["y"][s["tr"]])
        pred = clf.predict(s["F_own"][s["te"]])
        within[s["name"]] = report(s["name"], s["y"][s["te"]], pred)

    # ---- accuracy vs time gap
    s0 = sessions[0]
    print(f"\n2) ACCURACY-VS-TIME-GAP (train on {s0['name']} reps {a.train_reps}; "
          f"other sessions tested on {'all reps' if a.cross_test == 'all' else 'test reps'}):")
    clf0 = train_lda(s0["F_thr0"][s0["tr"]], s0["y"][s0["tr"]])
    rows = []
    for s in sessions:
        if s is s0 or a.cross_test == "test-reps":
            m = s["te"]
        else:
            m = np.ones(len(s["y"]), dtype=bool)
        pred = clf0.predict(s["F_thr0"][m])
        gap = (s["day"] - s0["day"]) + (s["sess"] - s0["sess"]) * 0.5
        b, fa = report(f"day{s['day']} T{s['sess']} (gap {gap:+.1f} d, n={int(m.sum())})", s["y"][m], pred)
        rows.append((s["name"], gap, b, fa))

    csv_path = out / "results_baseline.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "gap_days", "cross_session_balanced_acc", "cross_session_rest_false_act",
                    "within_session_balanced_acc", "within_session_rest_false_act"])
        for name, g, b, fa in rows:
            wb, wfa = within[name]
            w.writerow([name, f"{g:.1f}", f"{b:.4f}", f"{fa:.4f}", f"{wb:.4f}", f"{wfa:.4f}"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([r[1] for r in rows], [r[2] for r in rows], "o-", label="train on first session, test each session (drift curve)")
    ax.plot([r[1] for r in rows], [within[r[0]][0] for r in rows], "s--", color="gray", label="within-session ceiling")
    ax.set_xlabel("approx. days since training session")
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("DB6 baseline: accuracy vs time gap" + (f" (dropped ch {a.drop_channels})" if a.drop_channels else ""))
    fig.tight_layout()
    fig_path = out / "accuracy_vs_timegap.png"
    fig.savefig(fig_path, dpi=130)

    print(f"\nSaved:\n  {csv_path}\n  {fig_path}")
    print(f"""
How to read this:
  - WITHIN-SESSION = ceiling: plain features + LDA, no drift.
  - DRIFT CURVE = what the drift layer must beat. Its decay with the gap is the problem, measured.
  - rest-false-act = fraction of rest windows classified as a grasp (matters as much as accuracy).
  - Check the 'objects in train/test' lines before trusting any number; a WARNING there means
    the split confounds object with repetition.
  - Single subject, plain LDA, 7 grasps: a baseline for building the drift layer, not a result.
Done in {time.time() - t0:.0f}s.""")


if __name__ == "__main__":
    sys.exit(main())
