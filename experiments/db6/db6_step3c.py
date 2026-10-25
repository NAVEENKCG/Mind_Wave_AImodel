"""
db6_step3b.py - step 3b: fixed-pipeline re-test of the step-3 winner (mode B,
per-session feature z-score) plus the variants the step-3 review asked for.

The signal chain, windowing and features are copied VERBATIM from db6_baseline.py
v2 (causal 20-250 Hz 4th-order Butterworth, 2000 -> 1000 Hz, 200 ms windows at
100 ms step dropping label-change windows, MAV/WL/ZC/SSC per channel, ZC/SSC
threshold = 0.02 * per-channel std of the D1T1 train windows, rest windows inherit
the repetition they follow). Train = D1T1 reps 1-8 (LDA on raw features); the
train session's held-out reps 9-12 stay raw; every other session is tested on all
its reps. Identical train/test for every mode - only the feature transform differs.

Modes compared (all use the SAME frozen D1T1 classifier):
  baseline   raw features
  B          per-session z-score, stats from ALL windows of the target session
             (the step-3 winner; offline, uses the whole session)
  B_rest     z-score, stats from the session's REST windows only (does not
             depend on the session's grasp mix)
  B_train    z-score, stats from the target session's reps 1-8 only (removes
             the held-out-rep leak the review flagged)
  B_causal   z-score, stats from the FIRST --causal-windows windows of the
             session only (online-capable: a short calibration prefix)
  B_roll     past-only rolling z-score (exponential stats, --roll-halflife
             windows; online-capable, tracks drift inside the session)

Checks on top:
  - skewed-class-mix test: one grasp + rest only (--skew-session/--skew-grasp)
  - per-session confusion matrix with grasp->rest miss and grasp->grasp summary
  - repetition-level bootstrap CI on the accuracy gain vs baseline
  - 10x10 train-on-each-session matrix (baseline and B) to show drift structure
  - rest false activations as EVENTS PER MINUTE after causal majority smoothing

Install: same as v2 - pip install numpy scipy scikit-learn matplotlib
(matplotlib not needed here; h5py only if the .mat files are v7.3)

Usage (repo root, data from db6_explore.py in data/db6_subject):
  python experiments/db6/db6_step3b.py --dir data/db6_subject
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import scipy.io as sio

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
from scipy.signal import butter, sosfilt

FS_RAW = 2000
FS_OUT = 1000
BAND = (20.0, 250.0)
WIN_MS = 200
STEP_MS = 100
NAME_RE = re.compile(r"S(\d+)_D(\d+)_T(\d+)", re.IGNORECASE)

MODES = ["baseline", "B", "B_rest", "B_train", "B_causal", "B_roll"]
ONLINE = {"B_causal", "B_roll"}          # computable in real time, no future data
EPS = 1e-8


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


def train_lda(F, y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(F, y)
    return clf


# ------------------------------------------------------------- transforms
def zfit(F):
    mu = F.mean(axis=0)
    sd = F.std(axis=0)
    sd[sd < EPS] = EPS
    return mu, sd


def zapply(F, mu, sd):
    return (F - mu) / sd


def roll_zscore(F, halflife):
    """Past-only exponential z-score: window t is normalized with stats from
    windows before t (first window passes through raw stats of itself)."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    n, d = F.shape
    out = np.empty_like(F)
    mu = F[0].copy()
    var = np.zeros(d)
    out[0] = 0.0
    for t in range(1, n):
        out[t] = (F[t] - mu) / np.sqrt(var + EPS)
        diff = F[t] - mu
        mu += alpha * diff
        var = (1 - alpha) * (var + alpha * diff * diff)
    return out


def transform(mode, F, y, r, train_reps, causal_windows, halflife):
    if mode == "baseline":
        return F
    if mode == "B":
        return zapply(F, *zfit(F))
    if mode == "B_rest":
        m = y == 0
        return zapply(F, *zfit(F[m])) if m.any() else F
    if mode == "B_train":
        m = np.isin(r, train_reps)
        return zapply(F, *zfit(F[m])) if m.any() else F
    if mode == "B_causal":
        k = min(causal_windows, len(F))
        return zapply(F, *zfit(F[:k]))
    if mode == "B_roll":
        return roll_zscore(F, halflife)
    raise ValueError(mode)


# ---------------------------------------------------------------- metrics
def win_metrics(y, pred):
    from sklearn.metrics import balanced_accuracy_score
    acc = float((y == pred).mean())
    bacc = float(balanced_accuracy_score(y, pred))
    rest = y == 0
    rfa = float((pred[rest] != 0).mean()) if rest.any() else float("nan")
    return acc, bacc, rfa


def causal_majority(pred, k):
    """Causal majority vote over the last k windows (online-plausible smoothing)."""
    n = len(pred)
    out = pred.copy()
    lab_max = int(pred.max()) + 1
    for i in range(n):
        s = max(0, i - k + 1)
        out[i] = np.bincount(pred[s:i + 1], minlength=lab_max).argmax()
    return out


def rest_events_per_min(y, pred_sm):
    """False grasp activations per minute of true rest, after smoothing.
    An event is a run of consecutive rest windows smoothed-predicted as a grasp."""
    rest = np.flatnonzero(y == 0)
    if rest.size == 0:
        return float("nan")
    minutes = rest.size * STEP_MS / 60000.0
    fa = pred_sm[rest] != 0
    edges = int(np.sum(fa & ~np.concatenate(([False], fa[:-1]))))
    return edges / minutes if minutes > 0 else float("nan")


def g2r_miss(y, pred_sm):
    grasp = y > 0
    return float((pred_sm[grasp] == 0).mean()) if grasp.any() else float("nan")


def print_confusion(tag, y, pred, classes):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, pred, labels=classes).astype(float)
    row = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmn = cm / row
    names = ["rest"] + [f"g{int(c)}" for c in classes[1:]]
    print(f"    confusion {tag} (rows=true, cols=pred, row-normalized)")
    print("              " + " ".join(f"{n:>6s}" for n in names))
    for i, n in enumerate(names):
        print(f"      {n:>6s}  " + " ".join(f"{cmn[i, j]:6.2f}" for j in range(len(classes))))
    g = y > 0
    g2r = float((pred[g] == 0).mean()) if g.any() else float("nan")
    gi = [i for i, c in enumerate(classes) if c != 0]
    off = [cmn[i, j] for i in gi for j in gi if i != j]
    g2g = float(np.mean(off)) if off else float("nan")
    print(f"      grasp->rest miss {g2r:.3f} | grasp->grasp confusion (mean off-diag) {g2g:.3f}")
    return g2r, g2g


def boot_ci(preds, base_pred, ys, reps_list, n_boot, seed):
    """Repetition-level bootstrap of (mode acc - baseline acc), mean over sessions.
    Resamples repetitions within each session with replacement."""
    rng = np.random.default_rng(seed)
    diffs = {m: np.empty(n_boot) for m in preds if m != "baseline"}
    for b in range(n_boot):
        for m in diffs:
            vals = []
            for (y, r, pr, pb) in zip(ys, reps_list, preds[m], base_pred):
                u = np.unique(r)
                pick = rng.choice(u, size=len(u), replace=True)
                cnt = np.bincount(np.searchsorted(u, pick), minlength=len(u))   # times each repetition was drawn
                wt = cnt[np.searchsorted(u, r)].astype(float)
                vals.append(float((wt * (pr == y)).sum() / wt.sum()) - float((wt * (pb == y)).sum() / wt.sum()))
            diffs[m][b] = float(np.mean(vals))
    rows = []
    for m, d in diffs.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append((m, float(d.mean()), float(lo), float(hi), float((d > 0).mean())))
    return rows


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/db6_subject", help="folder with extracted .mat files")
    ap.add_argument("--out", default=None, help="output folder (default: <dir>/step3b)")
    ap.add_argument("--fs", type=float, default=FS_RAW)
    ap.add_argument("--drop-channels", nargs="*", type=int, default=[],
                    help="0-based channels to exclude, e.g. --drop-channels 7")
    ap.add_argument("--train-reps", default="1-8")
    ap.add_argument("--test-reps", default="9-12")
    ap.add_argument("--causal-windows", type=int, default=600,
                    help="prefix length for B_causal in windows (default 600 = 60 s)")
    ap.add_argument("--roll-halflife", type=int, default=300,
                    help="half-life of B_roll stats in windows (default 300 = 30 s)")
    ap.add_argument("--smooth", type=int, default=5, help="causal majority window in windows (default 5)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap resamples (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skew-session", default="D3T1", help="session id like D3T1 for the skew test")
    ap.add_argument("--skew-grasp", type=int, default=1, help="grasp label kept in the skew test")
    ap.add_argument("--cm-modes", default="baseline,B", help="comma list of modes for confusion matrices")
    ap.add_argument("--skip-matrix", action="store_true", help="skip the 10x10 train-on-each matrix")
    a = ap.parse_args()
    train_reps, test_reps = parse_reps(a.train_reps), parse_reps(a.test_reps)
    cm_modes = [m.strip() for m in a.cm_modes.split(",") if m.strip() in MODES]

    t0 = time.time()
    root = Path(a.dir)
    out = Path(a.out) if a.out else root / "step3b"
    out.mkdir(parents=True, exist_ok=True)

    def key(f):
        m = NAME_RE.search(f.name)
        return (int(m.group(2)), int(m.group(3)), str(f)) if m else (99, 99, str(f))

    files = sorted(root.rglob("*.mat"), key=key)
    files = [f for f in files if f.parent.name not in ("baseline", "step3b")]
    if not files:
        raise SystemExit(f"No .mat files under {root}.")
    print(f"Found {len(files)} .mat files under {root}  (sorted by day, session)")
    print(f"Train reps {train_reps} | test reps {test_reps} | ZC/SSC threshold 0.02 (v2, frozen on D1T1)")

    sessions, channels, thr0 = [], None, None
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

        X, y, r, o = make_windows(sig, lab, rep, obj, FS_OUT)
        del sig
        tr = np.isin(r, train_reps)
        te = np.isin(r, test_reps)
        ot = sorted(set(int(v) for v in np.unique(o[tr])) - {0})
        oe = sorted(set(int(v) for v in np.unique(o[te])) - {0})
        print(f"  loaded {f.name}: {len(y)} windows | objects train reps {ot} | test reps {oe}")
        if tr.sum() == 0 or te.sum() == 0:
            print(f"    WARNING: {f.name} has an empty train or test split")

        thr_own = zc_threshold(X[tr])
        if thr0 is None:
            thr0 = thr_own
            if (day, sess) != (1, 1):
                print(f"    WARNING: first session is D{day} T{sess}, not D1 T1")
        F = features(X, thr0)     # frozen train-session threshold for every session
        del X
        sessions.append(dict(name=f.name, day=day, sess=sess, y=y, r=r, tr=tr, te=te, F=F))

    if len(sessions) < 2:
        raise SystemExit("Need at least 2 usable sessions.")
    s0 = sessions[0]
    classes = np.unique(np.concatenate([s["y"] for s in sessions]))

    print(f"\nFitting one LDA per mode on {s0['name']} train reps ({int(s0['tr'].sum())} windows); each mode's "
          "classifier is trained on the SAME transform it applies to test sessions")
    F0, clfs = {}, {}
    for m in MODES:
        F0[m] = transform(m, s0["F"], s0["y"], s0["r"], train_reps, a.causal_windows, a.roll_halflife)
        clfs[m] = train_lda(F0[m][s0["tr"]], s0["y"][s0["tr"]])

    # ---- evaluate every mode with identical train/test
    preds = {m: [] for m in MODES}
    for s in sessions:
        for m in MODES:
            Ft = F0[m] if s is s0 else transform(m, s["F"], s["y"], s["r"], train_reps,
                                                 a.causal_windows, a.roll_halflife)
            preds[m].append(clfs[m].predict(Ft))   # full-length; tables apply the eval mask

    def sess_mask(s):
        return s["te"] if s is s0 else np.ones(len(s["y"]), dtype=bool)

    # ---- main table
    hdr = f"{'session':9s} " + " ".join(f"{m:>9s}" for m in MODES)
    print("\n1) ACCURACY % (train D1T1, * = same-session held-out reps)\n" + hdr)
    for i, s in enumerate(sessions):
        row = f"{s['name'].split('.')[0].split('_',1)[1] + ('*' if s is s0 else ''):9s} "
        mk = sess_mask(s)
        row += " ".join(f"{100*(preds[m][i][mk] == s['y'][mk]).mean():9.1f}" for m in MODES)
        print(row)
    cross = [i for i, s in enumerate(sessions) if s is not s0]
    mean_acc = {m: float(np.mean([(preds[m][i][sess_mask(sessions[i])] == sessions[i]["y"][sess_mask(sessions[i])]).mean() for i in cross])) for m in MODES}
    print(f"{'MEAN-x':9s} " + " ".join(f"{100*mean_acc[m]:9.1f}" for m in MODES) + "   (mean over the 9 other sessions)")

    print("\n2) BALANCED ACCURACY %\n" + hdr)
    from sklearn.metrics import balanced_accuracy_score
    for i, s in enumerate(sessions):
        mk = sess_mask(s)
        row = f"{s['name'].split('.')[0].split('_',1)[1] + ('*' if s is s0 else ''):9s} "
        row += " ".join(f"{100*balanced_accuracy_score(s['y'][mk], preds[m][i][mk]):9.1f}" for m in MODES)
        print(row)
    mean_b = {m: float(np.mean([balanced_accuracy_score(sessions[i]["y"][sess_mask(sessions[i])], preds[m][i][sess_mask(sessions[i])]) for i in cross])) for m in MODES}
    print(f"{'MEAN-x':9s} " + " ".join(f"{100*mean_b[m]:9.1f}" for m in MODES))

    print("\n3) REST FALSE ACTIVATIONS, per window %\n" + hdr)
    for i, s in enumerate(sessions):
        mk = sess_mask(s)
        row = f"{s['name'].split('.')[0].split('_',1)[1] + ('*' if s is s0 else ''):9s} "
        vals = []
        for m in MODES:
            y, p = s["y"][mk], preds[m][i][mk]
            rest = y == 0
            vals.append(100 * float((p[rest] != 0).mean()) if rest.any() else float("nan"))
        print(row + " ".join(f"{v:9.1f}" for v in vals))

    # ---- events per minute after causal majority smoothing
    print(f"\n4) REST FALSE EVENTS PER MINUTE after causal majority smoothing (k={a.smooth})  |  grasp->rest miss %\n" + hdr)
    for i, s in enumerate(sessions):
        mk = sess_mask(s)
        row = f"{s['name'].split('.')[0].split('_',1)[1] + ('*' if s is s0 else ''):9s} "
        ev, miss = [], []
        for m in MODES:
            p = causal_majority(preds[m][i][mk], a.smooth)
            ev.append(rest_events_per_min(s["y"][mk], p))
            miss.append(100 * g2r_miss(s["y"][mk], p))
        print(row + " ".join(f"{v:9.2f}" for v in ev))
        print(f"{'g2r-miss':9s} " + " ".join(f"{v:9.1f}" for v in miss))

    # ---- bootstrap CI on the gain
    print(f"\n5) BOOTSTRAP over repetitions (n={a.boot}): accuracy gain vs baseline, mean over the 9 cross sessions")
    base_cross = [preds["baseline"][i][sess_mask(sessions[i])] for i in cross]
    preds_cross = {m: [preds[m][i][sess_mask(sessions[i])] for i in cross] for m in MODES}
    ys_cross = [sessions[i]["y"][sess_mask(sessions[i])] for i in cross]
    reps_cross = [sessions[i]["r"][sess_mask(sessions[i])] for i in cross]
    for m, mu, lo, hi, ppos in boot_ci(preds_cross, base_cross, ys_cross, reps_cross, a.boot, a.seed):
        verdict = "significant gain" if lo > 0 else ("significant loss" if hi < 0 else "not significant")
        print(f"    {m:9s} gain {100*mu:+5.1f} pts  95% CI [{100*lo:+5.1f}, {100*hi:+5.1f}]  P(>0)={ppos:.2f}  -> {verdict}")

    # ---- skewed class mix
    print(f"\n6) SKEWED CLASS MIX: {a.skew_session}, only grasp {a.skew_grasp} + rest")
    mskw = re.match(r"D(\d+)T(\d+)", a.skew_session.strip().upper())
    sk = next((s for s in sessions if mskw and f"D{mskw.group(1)}_T{mskw.group(2)}" in s["name"]), None) if mskw else None
    if sk is None:
        print(f"    session {a.skew_session} not found, skipped")
    else:
        i = sessions.index(sk)
        msk = np.isin(sk["y"], [0, a.skew_grasp])
        print(f"    {'mode':9s} {'acc':>7s} {'bal':>7s} {'restFA%':>8s}   (n={int(msk.sum())} windows, grasp {a.skew_grasp} + rest)")
        for m in ["baseline", "B", "B_rest", "B_train", "B_causal"]:
            if sk is s0:
                Ft = F0[m][msk & sk["te"]]
                yy = sk["y"][msk & sk["te"]]
            else:   # the session only CONTAINS one grasp + rest: statistics come from that subset alone
                Ft = transform(m, sk["F"][msk], sk["y"][msk], sk["r"][msk], train_reps, a.causal_windows, a.roll_halflife)
                yy = sk["y"][msk]
            p = clfs[m].predict(Ft)
            acc, bacc, rfa = win_metrics(yy, p)
            print(f"    {m:9s} {100*acc:7.1f} {100*bacc:7.1f} {100*rfa:8.1f}")
        print("    B_rest is the only mode whose stats ignore the grasp mix. B, B_train and B_causal")
        print("    all include grasp windows in their stats, so a one-grasp session can break them.")

    # ---- confusion matrices
    print(f"\n7) CONFUSION MATRICES (modes: {', '.join(cm_modes)})")
    for i, s in enumerate(sessions):
        mk = sess_mask(s)
        print(f"  {s['name']}:")
        for m in cm_modes:
            print_confusion(m, s["y"][mk], preds[m][i][mk], classes)

    # ---- 10x10 train-on-each matrix
    if not a.skip_matrix:
        print("\n8) 10x10 TRAIN-ON-EACH-SESSION matrix, accuracy % (rows=train session, cols=test session;")
        print("   diagonal = held-out reps 9-12, off-diagonal = all reps)")
        names = [s["name"].split(".")[0].split("_", 1)[1] for s in sessions]
        for mode_name in ["baseline", "B"]:
            print(f"\n   mode = {mode_name}")
            print("          " + " ".join(f"{n:>6s}" for n in names))
            for i, si in enumerate(sessions):
                Fi = si["F"] if mode_name == "baseline" else zapply(si["F"], *zfit(si["F"]))
                clfi = train_lda(Fi[si["tr"]], si["y"][si["tr"]])
                row = []
                for j, sj in enumerate(sessions):
                    if j == i:
                        Ft, yy = Fi[sj["te"]], sj["y"][sj["te"]]
                    else:
                        Fj = sj["F"] if mode_name == "baseline" else zapply(sj["F"], *zfit(sj["F"]))
                        Ft, yy = Fj, sj["y"]
                    row.append(100 * float((clfi.predict(Ft) == yy).mean()))
                print(f"   {names[i]:>6s}  " + " ".join(f"{v:6.1f}" for v in row))

    # ---- csv summary
    csv_path = out / "step3b_summary.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "mode", "accuracy", "balanced", "rest_false_act_window"])
        for i, s in enumerate(sessions):
            mk = sess_mask(s)
            for m in MODES:
                acc, bacc, rfa = win_metrics(s["y"][mk], preds[m][i][mk])
                w.writerow([s["name"], m, f"{acc:.4f}", f"{bacc:.4f}", f"{rfa:.4f}"])
    print(f"\nSaved: {csv_path}")

    print(f"""
How to read this:
  - Identical train/test for every mode; only the feature transform changes.
  - baseline row should now match step 2 (threshold is the v2 0.02, frozen on D1T1):
    D1T2 about 41% accuracy and 4.3% rest-false-act. If it does, the pipeline is the step-2 one.
  - B vs B_train: the gap is the held-out-rep leak the review flagged.
  - B vs B_rest: the gap is grasp-mix dependence (check the skew test too).
  - B_causal / B_roll are the online-capable versions; the offline B is their ceiling.
  - Events/min and g2r-miss are after causal majority smoothing - the numbers a live hand would feel.
  - 10x10 matrix: look for day structure (blocks) vs session structure (checkerboard).
Done in {time.time() - t0:.0f}s.""")


if __name__ == "__main__":
    sys.exit(main())