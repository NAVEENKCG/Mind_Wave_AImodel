"""
Synthetic cued-calibration replay for NinaPro DB6 subject S1.

Calibration and evaluation are assembled from complete, contiguous
(grasp, repetition) blocks in the blocked recording.  This is a diagnostic,
not chronological live deployment: every synthetic block resets filter and
replay state, and no window crosses a join.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

try:
    from db6_replay import (
        BAND,
        EPS,
        FS_OUT,
        FS_RAW,
        STEP,
        THR_RATIO,
        WIN,
        WIN_SECONDS,
        bandpass_causal,
        causal_majority,
        col,
        decimate_slice,
        fit_lda,
        ffill_nonzero,
        load_mat,
        real_channels,
        session_windows,
        window_stack,
    )
except ImportError:
    from experiments.db6.db6_replay import (  # type: ignore
        BAND,
        EPS,
        FS_OUT,
        FS_RAW,
        STEP,
        THR_RATIO,
        WIN,
        WIN_SECONDS,
        bandpass_causal,
        causal_majority,
        col,
        decimate_slice,
        fit_lda,
        ffill_nonzero,
        load_mat,
        real_channels,
        session_windows,
        window_stack,
    )


METHODS = ("raw", "cued-stats", "cued-refit")
DEFAULT_KS = (0, 1, 2, 4)
DEFAULT_TAUS = (0.0, 0.6, 0.8)
RANDOM_SEED = 0


def discover_sessions(root):
    files = sorted(root.rglob("S1_D*_T*.mat"))
    if not files:
        raise SystemExit(f"No S1 session files found under {root}.")
    return files


def repetition_map(path):
    """Return actual nonzero repetition labels grouped by grasp."""
    d = load_mat(path)
    lab = col(d, "restimulus", "stimulus").astype(int)
    rep = col(d, "rerepetition", "repetition").astype(int)
    result = {}
    for grasp in sorted(np.unique(lab[lab > 0])):
        values = sorted(set(rep[lab == grasp][rep[lab == grasp] > 0].tolist()))
        result[int(grasp)] = values
    return result


def choose_split(files, max_k):
    """Choose one common split from the intersection of actual session labels."""
    maps = [repetition_map(path) for path in files]
    grasps = sorted(set.intersection(*(set(m) for m in maps)))
    if not grasps:
        raise ValueError("No grasp labels are shared by all sessions.")
    counts = {g: len(set.intersection(*(set(m[g]) for m in maps))) for g in grasps}
    if min(counts.values()) < max_k + 1:
        raise ValueError(f"Need at least {max_k + 1} common repetitions per grasp: {counts}")
    calibration_pool = {g: sorted(set.intersection(*(set(m[g]) for m in maps)))[:max_k]
                        for g in grasps}
    heldout = {g: sorted(set.intersection(*(set(m[g]) for m in maps)))[max_k:]
               for g in grasps}
    return calibration_pool, heldout, counts


def _feature_block(X, thr):
    mav = np.abs(X).mean(axis=1)
    dx = np.diff(X, axis=1)
    wl = np.abs(dx).sum(axis=1)
    zc = ((X[:, :-1] * X[:, 1:] < 0) & (np.abs(dx) >= thr)).sum(axis=1)
    a = X[:, 1:-1] - X[:, :-2]
    b = X[:, 1:-1] - X[:, 2:]
    ssc = ((a * b > 0) & ((np.abs(a) >= thr) | (np.abs(b) >= thr))).sum(axis=1)
    return np.hstack([mav, wl, zc, ssc])


def _features(sig, thr):
    n = 0 if len(sig) < WIN else (len(sig) - WIN) // STEP + 1
    if not n:
        return np.empty((0, 4 * sig.shape[1]))
    view = sliding_window_view(sig, WIN, axis=0)
    starts = np.arange(n) * STEP
    return _feature_block(view[starts].transpose(0, 2, 1), thr)


def _raw_metadata(path, fs):
    d = load_mat(path)
    emg = np.asarray(d["emg"])
    lab = col(d, "restimulus", "stimulus").astype(int)
    rep = ffill_nonzero(col(d, "rerepetition", "repetition"))
    obj = ffill_nonzero(col(d, "reobject", "object"))
    grasp = ffill_nonzero(lab)
    if fs != FS_RAW:
        raise ValueError("DB6 cued calibration currently requires --fs 2000.")
    return emg, lab, rep, obj, grasp


def repetition_blocks(path, channels, thr, fs=FS_RAW):
    """Build independent, filtered blocks, one for each contiguous grasp/repetition."""
    emg, lab, rep, obj, grasp = _raw_metadata(path, fs)
    key = grasp.astype(np.int64) * 1000 + rep
    changes = np.flatnonzero(np.r_[True, key[1:] != key[:-1], True])
    blocks = []
    q = int(round(fs / FS_OUT))
    for start, stop in zip(changes[:-1], changes[1:]):
        if grasp[start] <= 0 or rep[start] <= 0:
            continue
        raw = emg[start:stop, channels]
        sig = decimate_slice(bandpass_causal(raw, fs), fs)
        lab_s, rep_s, obj_s = lab[start:stop: q], rep[start:stop: q], obj[start:stop: q]
        n = 0 if len(sig) < WIN else (len(sig) - WIN) // STEP + 1
        if not n:
            continue
        starts = np.arange(n) * STEP
        windows = sliding_window_view(
            np.column_stack((lab_s, rep_s, obj_s)), WIN, axis=0
        )[starts].transpose(0, 2, 1)
        valid = np.all(windows == windows[:, :1, :], axis=(1, 2))
        y = lab_s[starts]
        F = _features(sig, thr)
        blocks.append(dict(
            session=Path(path).name,
            grasp=int(grasp[start]), rep=int(rep[start]), F=F, y=y, valid=valid,
            raw_start=int(start), raw_stop=int(stop),
            duration_seconds=(stop - start) / fs, original_start=start / fs,
            original_stop=stop / fs,
        ))
    return blocks


def select_blocks(blocks, repetitions):
    allowed = {(g, r) for g, reps in repetitions.items() for r in reps}
    return [b for b in blocks if (b["grasp"], b["rep"]) in allowed]


def calibration_durations(blocks):
    if not blocks:
        return 0.0, 0.0
    return (
        float(sum(b["duration_seconds"] for b in blocks)),
        float(max(b["original_stop"] for b in blocks) -
              min(b["original_start"] for b in blocks)),
    )


class IdentityStats:
    def transform(self, features):
        return features.copy()


class FrozenStats:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean).copy()
        self.std = np.asarray(std).copy()

    def transform(self, features):
        return (features - self.mean) / self.std


def fit_conditions(train_F, train_y, calibration_F, calibration_y):
    """Fit all three frozen conditions using no evaluation data."""
    raw = fit_lda(train_F, train_y)
    if len(calibration_F):
        mean = calibration_F.mean(axis=0)
        std = calibration_F.std(axis=0) + EPS
        stats = FrozenStats(mean, std)
        refit = fit_lda(np.vstack((train_F, calibration_F)),
                        np.concatenate((train_y, calibration_y)))
    else:
        stats = IdentityStats()
        refit = raw
    if not len(calibration_F):
        return {method: (raw, IdentityStats()) for method in METHODS}
    return {
        "raw": (raw, IdentityStats()),
        "cued-stats": (fit_lda((train_F - train_F.mean(axis=0)) /
                               (train_F.std(axis=0) + EPS), train_y), stats),
        "cued-refit": (refit, IdentityStats()),
    }


def _score_blocks(block_predictions, taus):
    from sklearn.metrics import balanced_accuracy_score

    results = {}
    for tau in taus:
        all_y, all_pred = [], []
        rest_count = grasp_count = false_rest = missed = events = 0
        rest_minutes = 0.0
        for block, proba, classes in block_predictions:
            pred = np.where(proba.max(axis=1) >= tau,
                            classes[proba.argmax(axis=1)], 0)
            mask = block["valid"]
            yt, yp = block["y"][mask], pred[mask]
            all_y.append(yt)
            all_pred.append(yp)
            rest = yt == 0
            grasp = yt > 0
            rest_count += int(rest.sum())
            grasp_count += int(grasp.sum())
            false_rest += int((yp[rest] != 0).sum())
            missed += int((yp[grasp] == 0).sum())
            settled = block["valid"] & (block["y"] == 0)
            settled_count = int(settled.sum())
            if settled_count:
                rest_minutes += settled_count * WIN_SECONDS / 60.0
            sm = causal_majority(pred, 5)
            fa = (sm != 0) & settled
            events += int(np.sum(fa[1:] & ~fa[:-1]) + (1 if len(fa) and fa[0] else 0))
        y = np.concatenate(all_y) if all_y else np.array([], dtype=int)
        p = np.concatenate(all_pred) if all_pred else np.array([], dtype=int)
        results[tau] = dict(
            acc=float((y == p).mean()) if len(y) else float("nan"),
            bal=float(balanced_accuracy_score(y, p)) if len(y) else float("nan"),
            rest_false=false_rest / rest_count if rest_count else float("nan"),
            miss=missed / grasp_count if grasp_count else float("nan"),
            events_min=events / max(rest_minutes, 1e-9),
            n=int(len(y)),
        )
    return results


def evaluate(blocks, fitted, taus):
    output = {}
    for method, (clf, corrector) in fitted.items():
        predictions = []
        for block in blocks:
            transformed = corrector.transform(block["F"])
            predictions.append((block, clf.predict_proba(transformed), clf.classes_))
        output[method] = _score_blocks(predictions, taus)
    return output


def _training_data(first, channels, fs):
    temp = session_windows(first, channels, None, fs)
    train_mask = temp["valid"] & np.isin(temp["rep"], np.arange(1, 9))
    X = window_stack(temp["sig"], np.flatnonzero(train_mask))
    thr = THR_RATIO * X.std(axis=(0, 1))
    session = session_windows(first, channels, thr, fs)
    mask = session["valid"] & np.isin(session["rep"], np.arange(1, 9))
    return session["F"][mask], session["y"][mask], thr


def run(root, ks=DEFAULT_KS, taus=DEFAULT_TAUS, out=None, fs=FS_RAW):
    files = discover_sessions(Path(root))
    max_k = max(ks)
    pool, heldout, counts = choose_split(files, max_k)
    print("Synthetic calibration/replay diagnostic; not chronological live deployment.")
    print(f"Random seed: {RANDOM_SEED} (no stochastic operation is used)")
    print("Actual repetitions per grasp (common across sessions):", counts)
    print("Split: calibration pool", pool, "| held-out", heldout)
    first_data = load_mat(files[0])
    channels = real_channels(first_data["emg"])
    train_F, train_y, thr = _training_data(files[0], channels, fs)
    rows = []
    for path in files[1:]:
        all_blocks = repetition_blocks(path, channels, thr, fs)
        eval_blocks = select_blocks(all_blocks, heldout)
        print(f"{path.name}: {len(all_blocks)} blocks, {sum(len(b['F']) for b in eval_blocks)} held-out windows")
        for k in ks:
            selected = {g: reps[:k] for g, reps in pool.items()}
            calibration = select_blocks(all_blocks, selected)
            cal_F = np.vstack([b["F"][b["valid"]] for b in calibration]) if calibration else np.empty((0, train_F.shape[1]))
            cal_y = np.concatenate([b["y"][b["valid"]] for b in calibration]) if calibration else np.empty(0, dtype=int)
            fitted = fit_conditions(train_F, train_y, cal_F, cal_y)
            durations = calibration_durations(calibration)
            results = evaluate(eval_blocks, fitted, taus)
            for method in METHODS:
                for tau, metrics in results[method].items():
                    rows.append(dict(k=k, method=method, tau=tau, session=path.name,
                                     **metrics, calib_seconds=durations[0],
                                     elapsed_original_seconds=durations[1]))
    destination = Path(out) if out else Path(root) / "cued" / "cued_results.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ["k", "method", "tau", "session", "acc", "bal", "rest_false", "miss",
              "events_min", "calib_seconds", "elapsed_original_seconds", "n"]
    with destination.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\nSummary: mean accuracy and calibration duration by session and k")
    for session in sorted({r["session"] for r in rows}):
        for k in ks:
            subset = [r for r in rows if r["session"] == session and r["k"] == k and r["tau"] == taus[0]]
            print(session, "k=", k, " ".join(f"{r['method']}={100*r['acc']:.1f}%/{r['calib_seconds']:.1f}s"
                                             for r in subset))
    print("\nAveraged across target sessions (tau=%.1f)" % taus[0])
    for k in ks:
        subset = [r for r in rows if r["k"] == k and r["tau"] == taus[0]]
        print("k=", k, " ".join(
            f"{method}={100*np.mean([r['acc'] for r in subset if r['method'] == method]):.1f}%/"
            f"{np.mean([r['calib_seconds'] for r in subset if r['method'] == method]):.1f}s"
            for method in METHODS
        ))
    print("Saved", destination)
    print("""
How to read this:
  one subject only; report per-session. This is a synthetic calibration/replay diagnostic,
  not chronological live deployment. calib_seconds is a lower-bound proxy from recorded
  block durations, not a measured calibration session. cued-stats uses labels only to select
  windows for statistics; cued-refit is supervised recalibration. False activations must be
  read next to missed grasps.""")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/db6_subject")
    parser.add_argument("--ks", nargs="*", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--taus", nargs="*", type=float, default=list(DEFAULT_TAUS))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args.dir, args.ks, args.taus, args.out)


if __name__ == "__main__":
    sys.exit(main())
