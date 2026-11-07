"""
db6_replay.py - causal replay harness for the NinaPro DB6 drift study.

Question it answers: can a correction that sees ONLY signals received so far (plus a
stated calibration budget) recover accuracy lost between sessions, and is the weak
causal result of step 3 real or an artefact of DB6's grasp-blocked recording order?

How it works
  - The classifier is trained once on session D1 T1 (reps 1-8) and frozen.
  - Every other session is REPLAYED window by window. A corrector may use the current
    window and earlier windows only. Two replay orders:
        original : windows in file order (DB6 repeats one grasp 12 times in a row)
        shuffled : complete repetitions (grasp + the rest that follows it) in random
                   order, never single windows. Several seeds.
    Shuffled replay is a DIAGNOSTIC (a user does not perform grasps in random order),
    not the real-world score. Report both.
  - Correctors:
        none      raw features (the baseline)
        offline   whole-session z-score. NOT causal: it sees future windows. Reference only.
        prefix    z-score from the first --warmup windows only, then frozen (causal)
        roll      exponentially weighted past-only z-score (causal)
  - The first --warmup windows of every replay are excluded from scoring for ALL methods,
    so all methods are scored on identical windows.
  - False events per minute: grasp outputs (after a 5-window causal majority) during SETTLED rest, i.e.
    more than 10 windows (1 s) after the last grasp window, so release delay is not counted as false.
  - Reject option: a prediction below confidence tau becomes REST. Report rest false
    activations TOGETHER WITH missed grasps (grasp windows answered REST), so rejecting
    everything cannot look safe.

Information budgets (state these in any report)
  none: nothing | offline: whole target session, unlabeled (non-causal) |
  prefix: first --warmup windows of the target session, unlabeled | roll: past windows, unlabeled.
  No target-session labels are used by any corrector.

Pipeline (same as db6_baseline.py v2): causal 4th-order 20-250 Hz Butterworth (sosfilt),
2000 -> 1000 Hz, 200 ms windows / 100 ms step, MAV, WL, ZC, SSC per channel, LDA,
ZC/SSC threshold = 2% of the training-window std, frozen. A window is SCORED only if label,
repetition and object are constant inside it. ALL windows are streamed.

Usage (repo root):
  python experiments/db6/db6_replay.py --dir data/db6_subject
  python experiments/db6/db6_replay.py --dir data/db6_subject --seeds 10 --warmup 600 --halflife 300

Install: pip install numpy scipy scikit-learn
Tests:   python -m pytest experiments/db6 -q
"""
import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfilt

FS_RAW = 2000
FS_OUT = 1000
BAND = (20.0, 250.0)
WIN = 200            # samples at FS_OUT (200 ms)
STEP = 100           # samples at FS_OUT (100 ms)
THR_RATIO = 0.02
TRAIN_REPS = list(range(1, 9))
TEST_REPS = list(range(9, 13))
EPS = 1e-12
WIN_SECONDS = STEP / FS_OUT
NAME_RE = re.compile(r"S(\d+)_D(\d+)_T(\d+)", re.IGNORECASE)


# ----------------------------------------------------------------- loading
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
    return [i for i, s in enumerate(emg.std(axis=0)) if s > tol]


def ffill_nonzero(v):
    """Zero samples take the last non-zero value before them; leading zeros take the first
    non-zero value (metadata only, never used for features)."""
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
def bandpass_causal(x, fs):
    sos = butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, x, axis=0)


def decimate_slice(x, fs):
    q = int(round(fs / FS_OUT))
    if abs(fs / q - FS_OUT) > 1e-6:
        raise ValueError(f"cannot downsample {fs} -> {FS_OUT} by an integer factor")
    return x[::q]


def n_windows(n_samples):
    return 0 if n_samples < WIN else (n_samples - WIN) // STEP + 1


def window_stack(sig, idx):
    """Windows (len(idx), WIN, C) for window numbers idx."""
    v = sliding_window_view(sig, WIN, axis=0)          # (T-WIN+1, C, WIN)
    return v[idx * STEP].transpose(0, 2, 1)


def _feat_block(X, thr):
    mav = np.abs(X).mean(axis=1)
    dx = np.diff(X, axis=1)
    wl = np.abs(dx).sum(axis=1)
    zc = ((X[:, :-1] * X[:, 1:] < 0) & (np.abs(dx) >= thr)).sum(axis=1)
    a = X[:, 1:-1] - X[:, :-2]
    b = X[:, 1:-1] - X[:, 2:]
    ssc = ((a * b > 0) & ((np.abs(a) >= thr) | (np.abs(b) >= thr))).sum(axis=1)
    return np.hstack([mav, wl, zc, ssc])


def features_all(sig, thr, chunk=800):
    """MAV, WL, ZC, SSC for EVERY window of sig (causal: window i uses samples
    [i*STEP, i*STEP+WIN) only). Returns (n, 4*C)."""
    n = n_windows(len(sig))
    out = [_feat_block(window_stack(sig, np.arange(i, min(i + chunk, n))), thr) for i in range(0, n, chunk)]
    return np.vstack(out) if out else np.empty((0, 4 * sig.shape[1]))


def stream_features(emg_raw, thr, fs=FS_RAW):
    """Raw (T, C) -> per-window features for every window. Used by the tests as well as
    by the harness, so the tests exercise the real code path."""
    sig = decimate_slice(bandpass_causal(emg_raw, fs), fs)
    return features_all(sig, thr)


def window_end_raw(n, fs=FS_RAW):
    """Raw-sample index (exclusive) at which window i has been completely received."""
    q = int(round(fs / FS_OUT))
    return (np.arange(n) * STEP + WIN) * q


# ------------------------------------------------------ session -> windows
def session_windows(path, channels, thr, fs=FS_RAW):
    """Load one DB6 file; return dict of per-window arrays for ALL windows."""
    d = load_mat(path)
    emg = d["emg"]
    lab = col(d, "restimulus", "stimulus").astype(int)
    rep = ffill_nonzero(col(d, "rerepetition", "repetition"))
    obj = ffill_nonzero(col(d, "reobject", "object"))
    gl = ffill_nonzero(lab)                              # grasp id carried through the rest after it
    sig = decimate_slice(bandpass_causal(emg[:, channels], fs), fs)
    q = int(round(fs / FS_OUT))
    lab, rep, obj, gl = lab[::q], rep[::q], obj[::q], gl[::q]
    n = n_windows(len(sig))
    starts = np.arange(n) * STEP
    ends = starts + WIN - 1

    def const(v):
        w = sliding_window_view(v, WIN)[starts]
        return w.min(axis=1) == w.max(axis=1)

    valid = const(lab) & const(rep) & const(obj)
    y = lab[starts]
    rep_w = rep[starts]
    key = gl[ends] * 1000 + rep[ends]                     # (grasp, repetition) of the window's last sample
    seg = np.concatenate([[0], np.cumsum(key[1:] != key[:-1])]) if n else np.array([], int)
    F = features_all(sig, thr) if thr is not None else None
    return dict(sig=sig, F=F, y=y, valid=valid, rep=rep_w, seg=seg, n=n)


# --------------------------------------------------------------- replay order
def replay_order(seg, mode, seed=0):
    """Order of window indices. 'shuffled' permutes COMPLETE segments (a grasp repetition
    plus the rest after it), never individual windows, and keeps each segment contiguous."""
    idx = np.arange(len(seg))
    if mode == "original" or len(seg) == 0:
        return idx
    if mode != "shuffled":
        raise ValueError(mode)
    n_seg = int(seg.max()) + 1
    groups = [np.flatnonzero(seg == s) for s in range(n_seg)]
    perm = np.random.default_rng(seed).permutation(n_seg)
    return np.concatenate([groups[i] for i in perm])


# ----------------------------------------------------------------- correctors
class Identity:
    name = "none"
    causal = True

    def transform(self, F):
        return F.copy()


class OfflineZ:
    """Whole-session z-score. Sees future windows: a non-causal REFERENCE only."""
    name = "offline"
    causal = False

    def transform(self, F):
        return (F - F.mean(axis=0)) / (F.std(axis=0) + EPS)


class PrefixZ:
    """z-score from the first n_warm windows, then frozen. Row i >= n_warm depends only on
    rows [0, n_warm) and row i. Rows before n_warm are returned unchanged (not scored)."""
    name = "prefix"
    causal = True

    def __init__(self, n_warm):
        self.n_warm = n_warm

    def transform(self, F):
        out = F.copy()
        if len(F) > self.n_warm:
            head = F[: self.n_warm]
            mu, sd = head.mean(axis=0), head.std(axis=0) + EPS
            out[self.n_warm:] = (F[self.n_warm:] - mu) / sd
        return out


class RollZ:
    """Exponentially weighted past-only z-score (expanding average at first, then half-life
    of `halflife` windows). The current window is included in its own statistics."""
    name = "roll"
    causal = True

    def __init__(self, halflife):
        self.alpha = 1.0 - 0.5 ** (1.0 / halflife)

    def transform(self, F):
        out = np.zeros_like(F, dtype=float)
        mu = np.zeros(F.shape[1])
        var = np.zeros(F.shape[1])
        for t in range(len(F)):
            a = max(self.alpha, 1.0 / (t + 1))
            d = F[t] - mu
            mu = mu + a * d
            var = (1 - a) * (var + a * d * d)
            out[t] = (F[t] - mu) / (np.sqrt(var) + EPS)
        return out


def make_correctors(warmup, halflife):
    return [Identity(), OfflineZ(), PrefixZ(warmup), RollZ(halflife)]


def is_causal(corrector, F, t, rng):
    """True iff changing rows AFTER t never changes outputs for rows <= t."""
    base = corrector.transform(F)[: t + 1]
    F2 = F.copy()
    F2[t + 1:] = rng.normal(size=F2[t + 1:].shape) * (np.abs(F).mean() + 1.0) * 50
    return bool(np.array_equal(base, corrector.transform(F2)[: t + 1]))


# -------------------------------------------------------------------- metrics
def causal_majority(pred, k=5):
    out = np.empty_like(pred)
    top = int(pred.max()) + 1 if len(pred) else 1
    for i in range(len(pred)):
        out[i] = np.bincount(pred[max(0, i - k + 1): i + 1], minlength=top).argmax()
    return out


def settled_rest(y, valid, order, guard):
    """True for valid REST windows that come at least `guard` windows after the last non-rest or
    invalid window in STREAM order. The majority smoother delays release by a few windows after a grasp
    ends; counting that delay as a 'false activation' would be wrong, so such windows are excluded."""
    ok = (valid & (y == 0))[order]
    out = np.zeros(len(order), bool)
    run = 0
    for i, flag in enumerate(ok):
        run = run + 1 if flag else 0
        out[i] = run > guard
    res = np.zeros(len(order), bool)
    res[order] = out
    return res


def score(proba, classes, y, order_pos, valid, warm, taus, guard=10):
    """proba, y, valid, order_pos are indexed by WINDOW number; order_pos[i] is the position of
    window i in the replay stream. Windows are scored only if valid and at stream position >= warm."""
    from sklearn.metrics import balanced_accuracy_score
    order = np.argsort(order_pos)                       # window numbers in stream order
    arg = classes[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    mask = valid & (order_pos >= warm)
    rest_scored = settled_rest(y, valid, order, guard) & (order_pos >= warm)
    minutes = max(rest_scored.sum() * WIN_SECONDS / 60.0, 1e-9)
    res = {}
    for tau in taus:
        pred = np.where(conf >= tau, arg, 0)            # below tau -> REST (the safe default)
        yt, yp = y[mask], pred[mask]
        rest, grasp = yt == 0, yt > 0
        sm = causal_majority(pred[order], 5)             # 5-window causal majority, in stream order
        fa = (sm != 0) & rest_scored[order]              # grasp output during SETTLED rest (>= guard windows after a grasp)
        events = int(np.sum(fa[1:] & ~fa[:-1]) + (1 if len(fa) and fa[0] else 0))
        res[tau] = dict(
            acc=float((yt == yp).mean()),
            bal=float(balanced_accuracy_score(yt, yp)),
            rest_false=float((yp[rest] != 0).mean()) if rest.any() else float("nan"),
            miss=float((yp[grasp] == 0).mean()) if grasp.any() else float("nan"),
            events_min=events / minutes,
            n=int(mask.sum()),
        )
    return res


def fit_lda(F, y):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
    clf.fit(F, y)
    return clf


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="data/db6_subject")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fs", type=float, default=FS_RAW)
    ap.add_argument("--warmup", type=int, default=600, help="windows excluded from scoring and used by 'prefix' (600 = 60 s)")
    ap.add_argument("--halflife", type=int, default=300, help="'roll' half-life in windows (300 = 30 s)")
    ap.add_argument("--seeds", type=int, default=5, help="shuffled-replay seeds")
    ap.add_argument("--taus", nargs="*", type=float, default=[0.0, 0.6, 0.8])
    a = ap.parse_args()

    t0 = time.time()
    root = Path(a.dir)
    out = Path(a.out) if a.out else root / "replay"
    out.mkdir(parents=True, exist_ok=True)

    def key(f):
        m = NAME_RE.search(f.name)
        return (int(m.group(2)), int(m.group(3)), str(f)) if m else (99, 99, str(f))

    files = sorted((f for f in root.rglob("*.mat") if "replay" not in f.parts), key=key)
    if len(files) < 2:
        raise SystemExit(f"Need at least 2 .mat files under {root}.")
    print(f"Found {len(files)} sessions; training session = {files[0].name}")
    m0 = NAME_RE.search(files[0].name)
    if m0 and (int(m0.group(2)), int(m0.group(3))) != (1, 1):
        print("WARNING: first session is not D1 T1")

    # ---- training session: channels, frozen feature threshold, frozen classifiers
    d0 = load_mat(files[0])
    channels = real_channels(d0["emg"])
    print(f"Channels ({len(channels)}): {channels}")
    tmp = session_windows(files[0], channels, None, a.fs)
    tr = tmp["valid"] & np.isin(tmp["rep"], TRAIN_REPS)
    Xtr = window_stack(tmp["sig"], np.flatnonzero(tr))
    thr = THR_RATIO * Xtr.std(axis=(0, 1))
    del Xtr
    s0 = session_windows(files[0], channels, thr, a.fs)
    tr = s0["valid"] & np.isin(s0["rep"], TRAIN_REPS)
    mu_tr, sd_tr = s0["F"][tr].mean(axis=0), s0["F"][tr].std(axis=0) + EPS
    clf_raw = fit_lda(s0["F"][tr], s0["y"][tr])
    clf_z = fit_lda((s0["F"][tr] - mu_tr) / sd_tr, s0["y"][tr])
    print(f"Classifiers fitted on {int(tr.sum())} windows of {files[0].name} reps {TRAIN_REPS[0]}-{TRAIN_REPS[-1]}")

    correctors = make_correctors(a.warmup, a.halflife)
    orders = [("original", 0)] + [("shuffled", s) for s in range(a.seeds)]
    rows = []                 # one row per (order, seed, method, tau, session)
    parity, ceiling = [], []

    for f in files[1:]:
        s = session_windows(f, channels, thr, a.fs)
        n = s["n"]
        print(f"  {f.name}: {n} windows, {int(s['valid'].sum())} scoreable, {int(s['seg'].max()) + 1} segments")
        # parity with earlier steps: raw features, all scoreable windows, no warm-up exclusion
        v = s["valid"]
        parity.append(float((clf_raw.predict(s["F"][v]) == s["y"][v]).mean()))
        # within-session ceiling for this session (own reps 1-8 -> 9-12, raw features)
        a_tr = v & np.isin(s["rep"], TRAIN_REPS)
        a_te = v & np.isin(s["rep"], TEST_REPS)
        thr_o = THR_RATIO * window_stack(s["sig"], np.flatnonzero(a_tr)).std(axis=(0, 1))
        Fo = features_all(s["sig"], thr_o)
        ceiling.append(float((fit_lda(Fo[a_tr], s["y"][a_tr]).predict(Fo[a_te]) == s["y"][a_te]).mean()))

        for oname, seed in orders:
            order = replay_order(s["seg"], oname, seed)
            pos = np.empty(n, int)
            pos[order] = np.arange(n)
            Fs = s["F"][order]                             # the stream
            for c in correctors:
                Fc = c.transform(Fs)
                clf = clf_raw if c.name == "none" else clf_z
                proba_stream = clf.predict_proba(Fc)
                proba = np.empty_like(proba_stream)
                proba[order] = proba_stream                # back to window index order
                res = score(proba, clf.classes_, s["y"], pos, s["valid"], a.warmup, a.taus)
                for tau, r in res.items():
                    rows.append(dict(order=oname, seed=seed, method=c.name, tau=tau,
                                     session=f.name, **r))

    # ---- summaries
    print(f"\nParity check: baseline accuracy on every scoreable window of the other sessions, "
          f"mean = {100 * np.mean(parity):.1f}%  (compare with step 2 / 3c; expect about 55.5% +/- 1)")
    print(f"Within-session reference (own reps 1-8 -> 9-12): mean = {100 * np.mean(ceiling):.1f}%  "
          "(a reference for each session, not a true ceiling)")

    sessions = sorted({r["session"] for r in rows})
    methods = [c.name for c in correctors]
    causal = {c.name: c.causal for c in correctors}

    def agg(order, method, tau, key_):
        per = {}
        for r in rows:
            if r["order"] == order and r["method"] == method and r["tau"] == tau:
                per.setdefault(r["session"], []).append(r[key_])
        vals = np.array([np.mean(per[ss]) for ss in sessions])
        return vals

    for order in ("original", "shuffled"):
        print(f"\n===== replay order: {order}"
              + (f"  (mean over {a.seeds} seeds; shuffles whole repetitions)" if order == "shuffled" else "")
              + f"  | warm-up {a.warmup} windows excluded for all methods =====")
        for tau in a.taus:
            print(f"\n  reject threshold tau = {tau:.1f}  (below tau -> REST)")
            print(f"  {'method':9s} {'causal':6s} {'acc%':>6s} {'bal%':>6s} {'rest-false%':>11s} {'missed%':>8s} "
                  f"{'false ev/min':>12s}  better-than-none")
            base = agg(order, "none", tau, "acc")
            for m in methods:
                acc, bal = agg(order, m, tau, "acc"), agg(order, m, tau, "bal")
                rf, ms, ev = agg(order, m, tau, "rest_false"), agg(order, m, tau, "miss"), agg(order, m, tau, "events_min")
                better = "-" if m == "none" else f"{int((acc > base).sum())}/{len(sessions)} sessions"
                print(f"  {m:9s} {'yes' if causal[m] else 'NO':6s} {100 * acc.mean():6.1f} {100 * bal.mean():6.1f} "
                      f"{100 * np.nanmean(rf):11.1f} {100 * np.nanmean(ms):8.1f} {np.nanmean(ev):12.2f}  {better}")

    if a.seeds > 1:
        print("\nSeed-to-seed spread of mean accuracy at tau=0 (shuffled order), a sanity check on the shuffling:")
        for m in methods:
            per_seed = []
            for sd in range(a.seeds):
                vals = [r["acc"] for r in rows if r["order"] == "shuffled" and r["seed"] == sd
                        and r["method"] == m and r["tau"] == 0.0]
                per_seed.append(np.mean(vals))
            print(f"  {m:9s} mean {100 * np.mean(per_seed):.1f}%  sd across seeds {100 * np.std(per_seed):.2f} pts")

    csv_path = out / "replay_results.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {csv_path}")
    print("""
How to read this:
  - 'offline' is the non-causal reference. 'prefix' and 'roll' are the causal candidates.
  - Compare ORIGINAL vs SHUFFLED for prefix/roll. If they recover in shuffled order but not in
    original order, the weak causal result was grasp-block order, not the idea. Shuffled order is a
    diagnostic, not the real-world score.
  - No corrector uses target-session labels. Say so, and keep supervised (cued) calibration separate.
  - Rest false activations must be read next to 'missed%': rejecting everything is not safe, just idle.
  - One subject: report per-session results, not a general claim.""")
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    sys.exit(main())
