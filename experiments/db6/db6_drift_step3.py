"""
db6_drift_step3.py - Step 3: label-free drift fixes tested against the step-2 baseline.
Personal project: Mind_Wave_AImodel (hybrid EEG-EMG prosthetic hand). NinaPro DB6, one subject.

Question: train on Day 1 morning (D1T1, reps 1-8). Test on every session. Can we pull the
drifted-session accuracy up toward the ~70% ceiling WITHOUT using any gesture labels from the
new session?

Modes compared (same train data, same test sessions, same classifier = LDA):
  baseline  : unchanged. Filter, windows, MAV/WL/ZC/SSC, LDA.
  A         : divide each channel by the NEW session's own rest level, found WITHOUT labels
              (10th percentile of window MAV per channel = quietest windows).
  A_oracle  : like A, but rest level from the session's true rest windows (uses the rest label
              only, no gesture labels). Upper bound for A. Not label-free.
  B         : z-score every feature with that session's own unlabeled mean/std.
  C         : A plus automatic dead-channel mask (label-free, see --dead-ratio, --dead-spread).
  D         : C plus B (extra combination).

Reported per session: accuracy, balanced accuracy, rest false-activation (true rest windows
predicted as a gesture). Same-session row (D1T1) is tested on reps 9-12 only.

Separate experiment (not a default change):
  --filter-test   re-runs baseline, A and C with other band-pass / notch settings.
  --band LOW HIGH and --notch HZ [HZ ...] set one custom filter for the main run.

Install:  pip install numpy scipy scikit-learn matplotlib   (h5py only if a file is v7.3)
Run:      python experiments\\db6\\db6_drift_step3.py --dir data\\db6_subject
Output:   data\\db6_subject\\step3\\  (summary.csv, step3_accuracy.png, step3_rest_false.png)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, iirnotch, sosfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import balanced_accuracy_score

NAME_RE = re.compile(r"S(\d+)_D(\d+)_T(\d+)", re.I)
FS_RAW = 2000.0
FS = 1000.0
WIN = 200          # 200 ms at 1000 Hz
STEP = 100         # 100 ms
MODES = ["baseline", "A", "A_oracle", "B", "C", "D"]
MODE_NOTE = {"baseline": "unchanged", "A": "rest-level norm (label-free)",
             "A_oracle": "rest-level norm (true rest, upper bound)", "B": "per-session feature z-score",
             "C": "A + dead-channel mask", "D": "C + B"}
THR_RATIO = 0.05   # ZC/SSC threshold = this x channel std in the training session


# ------------------------------------------------------------------ loading
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


def find_files(root):
    files = []
    for f in Path(root).rglob("*.mat"):
        m = NAME_RE.search(f.name)
        if m:
            files.append((int(m.group(2)), int(m.group(3)), f))
    files.sort(key=lambda t: (t[0], t[1]))
    return files


def parse_reps(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


# ------------------------------------------------------------------ filtering and windows
def make_filters(low, high, notch_hz):
    sos = [butter(4, [low, high], btype="bandpass", fs=FS_RAW, output="sos")]
    for f0 in notch_hz:
        b, a = iirnotch(f0, 30.0, fs=FS_RAW)
        from scipy.signal import tf2sos
        sos.append(tf2sos(b, a))
    return np.vstack(sos)


def fill_reps(rep):
    """Rest carries rep 0. Give rest the repetition it follows (leading rest: the one it precedes)."""
    rep = rep.astype(np.int64).copy()
    nz = np.nonzero(rep)[0]
    if len(nz) == 0:
        return rep
    idx = np.searchsorted(nz, np.arange(len(rep)), side="right") - 1
    out = np.where(idx >= 0, rep[nz[np.clip(idx, 0, None)]], rep[nz[0]])
    return out


def window_meta(lab, rep, obj, nwin_from):
    """Per-window label/rep/object; keep = window has one label and one repetition."""
    starts = np.arange(0, len(lab) - WIN + 1, STEP)
    lw = sliding_window_view(lab, WIN)[starts]
    rw = sliding_window_view(rep, WIN)[starts]
    keep = (lw.min(1) == lw.max(1)) & (rw.min(1) == rw.max(1))
    ow = None
    if obj is not None:
        ow_windows = sliding_window_view(obj, WIN)[starts]
        keep &= ow_windows.min(1) == ow_windows.max(1)
        ow = ow_windows[:, -1]
    return starts, lw[:, -1], rw[:, -1], keep, ow


def window_feats(x, starts, thr):
    """x: (T, C) float32. Returns (nW, 4*C): MAV, WL, ZC, SSC (textbook definitions)."""
    nC = x.shape[1]
    out = np.empty((len(starts), 4 * nC), dtype=np.float64)
    view = sliding_window_view(x, WIN, axis=0)          # (T-WIN+1, C, WIN)
    for i in range(0, len(starts), 800):
        w = view[starts[i:i + 800]].astype(np.float64)  # (n, C, WIN)
        mav = np.abs(w).mean(-1)
        d = np.diff(w, axis=-1)
        wl = np.abs(d).sum(-1)
        t = thr[None, :, None]
        zc = ((w[..., :-1] * w[..., 1:] < 0) & (np.abs(d) >= t)).sum(-1)
        d1, d2 = d[..., :-1], d[..., 1:]
        ssc = ((-d1 * d2) >= t ** 2).sum(-1)             # (x_k - x_k-1)*(x_k - x_k+1) >= thr^2
        out[i:i + 800] = np.concatenate([mav, wl, zc, ssc], axis=1)
    return out


def mav_only(x, starts):
    view = sliding_window_view(np.abs(x), WIN, axis=0)
    out = np.empty((len(starts), x.shape[1]))
    for i in range(0, len(starts), 800):
        out[i:i + 800] = view[starts[i:i + 800]].mean(-1)
    return out


def extract_session(path, sos, ref, used_ref=None):
    """Load one session. Returns dict with features for each signal variant and window metadata."""
    d = load_mat(path)
    emg = np.asarray(d["emg"], dtype=np.float64)
    used = np.where(emg.std(0) > 1e-9)[0] if used_ref is None else used_ref
    x = sosfilt(sos, emg[:, used], axis=0)[::2].astype(np.float32)    # causal filter, 2000 -> 1000 Hz
    del emg
    lab = col(d, "restimulus", "stimulus")[::2].astype(np.int64)
    rep0 = col(d, "rerepetition", "repetition")[::2]
    rep = fill_reps(rep0)
    obj = col(d, "reobject", "object")
    obj = None if obj is None else obj[::2].astype(np.int64)
    n = min(len(x), len(lab), len(rep), len(obj) if obj is not None else 10 ** 12)
    x, lab, rep = x[:n], lab[:n], rep[:n]
    obj = None if obj is None else obj[:n]
    starts, wl, wr, keep, wo = window_meta(lab, rep, obj, n)
    starts, wl, wr = starts[keep], wl[keep], wr[keep]
    wo = None if wo is None else wo[keep]

    m_raw = mav_only(x, starts)                                         # (nW, C)
    g_proxy = np.percentile(m_raw, 10, axis=0)                          # label-free rest level
    rest = wl == 0
    g_orac = np.median(m_raw[rest], axis=0) if rest.any() else g_proxy
    spread = np.percentile(m_raw, 90, axis=0) / np.maximum(np.percentile(m_raw, 10, axis=0), 1e-12)

    if ref is None:   # first (training) session sets thresholds for each signal variant
        ref = {"thr_raw": THR_RATIO * x.std(0),
               "thr_A": THR_RATIO * (x / g_proxy).std(0),
               "thr_O": THR_RATIO * (x / g_orac).std(0)}
    f = {"raw": window_feats(x, starts, ref["thr_raw"]),
         "A": window_feats((x / np.maximum(g_proxy, 1e-12)).astype(np.float32), starts, ref["thr_A"]),
         "O": window_feats((x / np.maximum(g_orac, 1e-12)).astype(np.float32), starts, ref["thr_O"])}
    return {"feats": f, "label": wl, "rep": wr, "obj": wo, "g_proxy": g_proxy, "g_orac": g_orac,
            "spread": spread, "used": used}, ref


# ------------------------------------------------------------------ modelling
def zscore(F):
    return (F - F.mean(0)) / np.maximum(F.std(0), 1e-12)


def feat_cols(chs, nC):
    return np.concatenate([chs + k * nC for k in range(4)])


def fit_eval(Ftr, ytr, Fte, yte, cols=None):
    if cols is not None:
        Ftr, Fte = Ftr[:, cols], Fte[:, cols]
    clf = LinearDiscriminantAnalysis().fit(Ftr, ytr)
    p = clf.predict(Fte)
    rest = yte == 0
    return (float((p == yte).mean() * 100), float(balanced_accuracy_score(yte, p) * 100),
            float((p[rest] != 0).mean() * 100) if rest.any() else float("nan"))


def run(files, sos, args, drop, label=""):
    train_reps, test_reps = parse_reps(args.train_reps), parse_reps(args.test_reps)
    ref, tr, used = None, None, None
    sessions = []
    for day, ses, path in files:
        s, ref = extract_session(path, sos, ref, used)
        if used is None:
            used = s["used"]
        s["name"] = f"D{day}T{ses}"
        s["path"] = path
        sessions.append(s)
        print(f"  loaded {path.name}: {len(s['label'])} windows", flush=True)
    tr = sessions[0]
    nC = len(used)
    keep_ch = np.array([i for i, c in enumerate(used) if c not in drop])
    ttr = np.isin(tr["rep"], train_reps)
    ytr = tr["label"][ttr]

    # objects check (train vs test reps of the training session)
    if tr["obj"] is not None and not label:
        a = sorted(set(tr["obj"][ttr].tolist()))
        b = sorted(set(tr["obj"][np.isin(tr["rep"], test_reps)].tolist()))
        print(f"  objects train reps: {a}\n  objects test reps:  {b}")
        if set(a) != set(b):
            print("  WARNING: objects differ between train and test reps - try --train-reps 1-6 --test-reps 7-12")

    gp_tr = tr["g_proxy"]
    rows = []
    for s in sessions:
        same = s is tr
        sel = np.isin(s["rep"], test_reps) if same else np.ones(len(s["label"]), bool)
        yte = s["label"][sel]
        ratio = s["g_proxy"] / gp_tr
        dead = (ratio < args.dead_ratio) | (s["spread"] < args.dead_spread)
        dead_tr = (tr["spread"] < args.dead_spread)
        live = np.array([c for c in keep_ch if not dead[c] and not dead_tr[c]])
        if len(live) < 4:
            live = keep_ch
        res = {}
        kc_all = feat_cols(keep_ch, nC)
        kc_live = feat_cols(live, nC)
        for mode in MODES:
            if mode == "baseline":
                Ftr, Fte, cols = tr["feats"]["raw"][ttr], s["feats"]["raw"][sel], kc_all
            elif mode == "A":
                Ftr, Fte, cols = tr["feats"]["A"][ttr], s["feats"]["A"][sel], kc_all
            elif mode == "A_oracle":
                Ftr, Fte, cols = tr["feats"]["O"][ttr], s["feats"]["O"][sel], kc_all
            elif mode == "B":
                Ftr_all, Fte_all = zscore(tr["feats"]["raw"]), zscore(s["feats"]["raw"])
                Ftr, Fte, cols = Ftr_all[ttr], Fte_all[sel], kc_all
            elif mode == "C":
                Ftr, Fte, cols = tr["feats"]["A"][ttr], s["feats"]["A"][sel], kc_live
            else:  # D
                Ftr_all, Fte_all = zscore(tr["feats"]["A"]), zscore(s["feats"]["A"])
                Ftr, Fte, cols = Ftr_all[ttr], Fte_all[sel], kc_live
            res[mode] = fit_eval(Ftr, ytr, Fte, yte, cols)
        masked = [int(used[c]) for c in keep_ch if c not in live]
        lr = np.abs(np.log(s["g_proxy"] / np.maximum(s["g_orac"], 1e-12)))
        rows.append({"session": s["name"], "same": same, "res": res, "masked": masked,
                     "proxy_vs_oracle": float(np.median(lr))})
    return rows, used


# ------------------------------------------------------------------ reporting
def print_tables(rows, modes=MODES):
    for idx, title in [(0, "ACCURACY %"), (1, "BALANCED ACCURACY %"), (2, "REST FALSE ACTIVATION % (lower is better)")]:
        print(f"\n{title}")
        print("session   " + "".join(f"{m:>10}" for m in modes))
        for r in rows:
            tag = r["name"] if "name" in r else r["session"]
            print(f"{tag + ('*' if r['same'] else ''):<10}" + "".join(f"{r['res'][m][idx]:10.1f}" for m in modes))
    print("  (* = same session as training, tested on held-out reps only)")
    cross = [r for r in rows if not r["same"]]
    print("\nMEAN OVER THE OTHER SESSIONS (drift test)")
    print("mode        acc   bal.acc   rest-false")
    for m in modes:
        a = np.mean([r["res"][m][0] for r in cross])
        b = np.mean([r["res"][m][1] for r in cross])
        c = np.nanmean([r["res"][m][2] for r in cross])
        print(f"{m:<10}{a:6.1f}{b:9.1f}{c:12.1f}   {MODE_NOTE.get(m, '')}")
    print("\nDEAD-CHANNEL MASK (mode C/D) per session, DB6 channel index:")
    for r in rows:
        print(f"  {r['session']}: masked {r['masked'] if r['masked'] else 'none'}   "
              f"| proxy-vs-true rest level, median |log ratio| = {r['proxy_vs_oracle']:.2f} (0 = identical)")


def save_outputs(rows, outdir):
    import csv
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session", "same_session", "mode", "accuracy", "balanced_accuracy", "rest_false_activation", "masked_channels"])
        for r in rows:
            for m in MODES:
                a, b, c = r["res"][m]
                w.writerow([r["session"], r["same"], m, f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", " ".join(map(str, r["masked"]))])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing - skipped plots")
        return
    names = [r["session"] for r in rows]
    x = np.arange(len(rows))
    for idx, fn, yl in [(0, "step3_accuracy.png", "accuracy %"), (2, "step3_rest_false.png", "rest false activation %")]:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for m in MODES:
            ax.plot(x, [r["res"][m][idx] for r in rows], marker="o", lw=2.2 if m == "baseline" else 1.3,
                    ls="--" if m == "A_oracle" else "-", label=m, color="k" if m == "baseline" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(yl)
        ax.set_xlabel("test session (training = D1T1 reps 1-8)")
        ax.grid(alpha=.3)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / fn, dpi=130)
        plt.close(fig)
    print(f"\nSaved: {outdir / 'summary.csv'}, step3_accuracy.png, step3_rest_false.png")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder with the S#_D#_T#.mat files")
    ap.add_argument("--out", default=None, help="output folder (default: <dir>/step3)")
    ap.add_argument("--drop-channels", nargs="*", type=int, default=[], help="DB6 channel indices to drop")
    ap.add_argument("--train-reps", default="1-8")
    ap.add_argument("--test-reps", default="9-12")
    ap.add_argument("--dead-ratio", type=float, default=0.2,
                    help="mask a channel if its rest level < this x the training session's (default 0.2)")
    ap.add_argument("--dead-spread", type=float, default=1.5,
                    help="mask a channel if loud/quiet window ratio (p90/p10 MAV) < this (default 1.5)")
    ap.add_argument("--band", nargs=2, type=float, default=[20, 250], metavar=("LOW", "HIGH"),
                    help="band-pass in Hz for the main run (default 20 250 = baseline)")
    ap.add_argument("--notch", nargs="*", type=float, default=[], help="optional notch frequencies in Hz")
    ap.add_argument("--filter-test", action="store_true",
                    help="EXPERIMENT: also compare other filters (20-200, 20-150, notch 74/148/222)")
    a = ap.parse_args()

    files = find_files(a.dir)
    if not files:
        raise SystemExit("No S#_D#_T#.mat files found.")
    print(f"Found {len(files)} sessions: " + ", ".join(f"D{d}T{s}" for d, s, _ in files))
    if (files[0][0], files[0][1]) != (1, 1):
        print("WARNING: D1T1 not found - using the first session as training.")
    print(f"Filter: {a.band[0]:g}-{a.band[1]:g} Hz causal 4th order" + (f" + notch {a.notch}" if a.notch else ""))
    print("Train session:", files[0][2].name, "reps", a.train_reps, "| test reps (same session)", a.test_reps)

    sos = make_filters(a.band[0], a.band[1], a.notch)
    rows, used = run(files, sos, a, set(a.drop_channels))
    print(f"Channels used: {used.tolist()}" + (f" (dropped {a.drop_channels})" if a.drop_channels else ""))
    print_tables(rows)
    out = Path(a.out) if a.out else Path(a.dir) / "step3"
    save_outputs(rows, out)

    if a.filter_test:
        print("\n=== FILTER EXPERIMENT (separate from the main result) ===")
        cfgs = [("20-250", (20, 250), []), ("20-200", (20, 200), []), ("20-150", (20, 150), []),
                ("20-250+notch74/148/222", (20, 250), [74, 148, 222])]
        summ = []
        for name, (lo, hi), nt in cfgs:
            print(f"\n-- {name}", flush=True)
            r, _ = run(files, make_filters(lo, hi, nt), a, set(a.drop_channels), label=name)
            cross = [x for x in r if not x["same"]]
            same = [x for x in r if x["same"]]
            summ.append((name, {m: (np.mean([x["res"][m][0] for x in cross]), np.mean([x["res"][m][2] for x in cross]),
                                    same[0]["res"][m][0]) for m in ("baseline", "A", "C")}))
        print("\nFILTER EXPERIMENT SUMMARY: mean accuracy / mean rest-false over the other sessions, [same-session acc]")
        print(f"{'filter':<26}{'baseline':>22}{'A':>22}{'C':>22}")
        for name, d in summ:
            print(f"{name:<26}" + "".join(f"{d[m][0]:8.1f}/{d[m][1]:4.1f} [{d[m][2]:4.1f}]  " for m in ("baseline", "A", "C")))
        print("Note: 74/148/222 Hz notches follow the 222/296/370/444 Hz peaks seen in step 1 (a 74 Hz harmonic series).")
        print("A filter that wins here is a hypothesis to confirm, not a proven fix of the interference source.")


if __name__ == "__main__":
    main()
