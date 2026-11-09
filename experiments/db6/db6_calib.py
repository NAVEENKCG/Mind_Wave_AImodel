"""Label-budget calibration audit for NinaPro DB6."""
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from db6_replay import (  # noqa: E402
    EPS,
    FS_RAW,
    NAME_RE,
    TEST_REPS,
    THR_RATIO,
    TRAIN_REPS,
    WIN_SECONDS,
    causal_majority,
    ffill_nonzero,
    fit_lda,
    score,
    session_windows,
    settled_rest,
    window_stack,
)


@dataclass(frozen=True)
class Split:
    calibration_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def cal_idx(self):
        return self.calibration_idx


def make_split(rep, valid, k):
    rep, valid = np.asarray(rep), np.asarray(valid, dtype=bool)
    calibration = valid & np.isin(rep, np.arange(1, k + 1))
    test = valid & np.isin(rep, TEST_REPS)
    if np.intersect1d(np.flatnonzero(calibration), np.flatnonzero(test)).size:
        raise AssertionError("calibration and test windows overlap")
    return Split(np.flatnonzero(calibration), np.flatnonzero(test))


def fit_z_stats(X, test_X=None):
    del test_X
    X = np.asarray(X, dtype=float)
    return X.mean(axis=0), X.std(axis=0) + EPS


def z_apply(X, stats):
    mu, sd = stats
    return (np.asarray(X, dtype=float) - mu) / sd


def fit_lda_model(X, y):
    return fit_lda(np.asarray(X), np.asarray(y))


def predict_arm(model, X, labels=None):
    """Predict from features only; labels is accepted solely for a leakage test."""
    del labels
    return model.predict(np.asarray(X))


def segment_table(y, rep, seg):
    rows = []
    counts = {}
    for sid in np.unique(seg):
        ix = np.flatnonzero(seg == sid)
        active = y[ix][y[ix] > 0]
        key = (int(active[0]) if len(active) else 0, int(rep[ix[0]]))
        counts[key] = counts.get(key, 0) + 1
    for sid in np.unique(seg):
        ix = np.flatnonzero(seg == sid)
        active = y[ix][y[ix] > 0]
        key = (int(active[0]) if len(active) else 0, int(rep[ix[0]]))
        if not len(active):
            rows.append({"segment": int(sid), "grasp": 0, "rep": int(rep[ix[0]]),
                         "n_windows": len(ix), "fragments": counts[key]})
        else:
            rows.append({
                "segment": int(sid),
                "grasp": int(active[0]),
                "rep": int(rep[ix[0]]),
                "n_windows": len(ix),
                "fragments": counts[key],
            })
    return rows


def metrics(proba, classes, y, valid, order_pos, warm=0):
    result = score(proba, classes, y, order_pos, valid, warm, [0.0])[0.0]
    pred = classes[np.asarray(proba).argmax(axis=1)]
    mask = valid & (order_pos >= warm)
    grasp = mask & (y > 0)
    wrong = grasp & (pred > 0) & (pred != y)
    recalls = {}
    for label in sorted(np.unique(y[mask])):
        selected = mask & (y == label)
        recalls[int(label)] = float(np.mean(pred[selected] == label)) if selected.any() else float("nan")
    return {
        "accuracy": result["acc"],
        "balanced_accuracy": result["bal"],
        "rest_false_pct": 100.0 * result["rest_false"],
        "missed_grasp_pct": 100.0 * result["miss"],
        "wrong_grasp_pct": 100.0 * (wrong.sum() / max(grasp.sum(), 1)),
        "false_events_min": result["events_min"],
        "per_class_recall": recalls,
        "n": result["n"],
    }


def better_than_none(rows, metric, higher):
    sessions = sorted({row["session"] for row in rows})
    better = 0
    for session in sessions:
        baseline = [float(row[metric]) for row in rows
                    if row["session"] == session and row["arm"] == "none" and int(row["k"]) == 0]
        candidate = [float(row[metric]) for row in rows
                     if row["session"] == session and row["arm"] != "none"]
        if not baseline or not candidate:
            continue
        b, c = np.mean(baseline), np.mean(candidate)
        better += bool(c > b) if higher else bool(c < b)
    return better


def _session_files(root):
    def key(path):
        match = NAME_RE.search(path.name)
        return (int(match.group(2)), int(match.group(3)), str(path)) if match else (99, 99, str(path))
    return sorted((p for p in Path(root).rglob("*.mat") if "replay" not in p.parts), key=key)


def _prepare(root):
    from db6_replay import load_mat, real_channels

    files = _session_files(root)
    if len(files) != 10:
        raise SystemExit(f"Expected D1 T1 plus 9 sessions, found {len(files)} under {root}.")
    channels = real_channels(load_mat(files[0])["emg"])
    raw = session_windows(files[0], channels, None, FS_RAW)
    train_mask = raw["valid"] & np.isin(raw["rep"], TRAIN_REPS)
    thr = THR_RATIO * window_stack(raw["sig"], np.flatnonzero(train_mask)).std(axis=(0, 1))
    return files, channels, thr, session_windows(files[0], channels, thr)


def _models(train, calibration, k):
    train_mask = train["valid"] & np.isin(train["rep"], TRAIN_REPS)
    if k:
        cal_mask = calibration["valid"] & np.isin(calibration["rep"], np.arange(1, k + 1))
        cal_X, cal_y = calibration["F"][cal_mask], calibration["y"][cal_mask]
        new = fit_lda_model(cal_X, cal_y)
        pooled = fit_lda_model(np.vstack([train["F"][train_mask], cal_X]),
                               np.concatenate([train["y"][train_mask], cal_y]))
        d1_stats = fit_z_stats(train["F"][train_mask])
        cal_stats = fit_z_stats(cal_X)
        pooled_z = fit_lda_model(
            np.vstack([z_apply(train["F"][train_mask], d1_stats), z_apply(cal_X, cal_stats)]),
            np.concatenate([train["y"][train_mask], cal_y]),
        )
        d1_z = fit_lda_model(z_apply(train["F"][train_mask], d1_stats), train["y"][train_mask])
        return {"new_only": new, "pooled": pooled, "pooled_z": (pooled_z, cal_stats), "cal_z_only": (d1_z, cal_stats)}
    return {}


def run_experiment(root, output):
    files, channels, thr, train = _prepare(root)
    print(f"Training session: {files[0].name}; replay sessions: {len(files) - 1}")
    table_session = session_windows(files[1], channels, thr)
    table = segment_table(table_session["y"], table_session["rep"], table_session["seg"])
    print(f"\nSEGMENT TABLE: {files[1].stem}")
    print("segment grasp repetition n_windows fragments")
    for row in table:
        print(f"{row['segment']:7d} {row['grasp']:6d} {row['rep']:11d}"
              f" {row['n_windows']:9d} {row['fragments']:9d}")
    expected = 7 * 12
    fragments = [r for r in table if r["fragments"] > 1]
    print(f"Segments={len(table)}; expected={expected}; extra={len(table) - expected}")
    print(f"Fragments (repeated grasp, repetition keys): {fragments}")

    train_mask = train["valid"] & np.isin(train["rep"], TRAIN_REPS)
    raw_model = fit_lda_model(train["F"][train_mask], train["y"][train_mask])
    train_seconds = {k: [] for k in (0, 1, 2, 4)}
    rows = []
    arms = ("none", "new_only", "pooled", "pooled_z", "cal_z_only", "reference")
    for path in files[1:]:
        s = session_windows(path, channels, thr)
        print(f"\n{path.stem}:")
        for k in (0, 1, 2, 4):
            split = make_split(s["rep"], s["valid"], k)
            train_seconds[k].append(len(split.cal_idx) * WIN_SECONDS)
            print(f"  k={k}: calibration={len(split.cal_idx) * WIN_SECONDS:.1f}s"
                  f" test_windows={len(split.test_idx)}")
            cal = s
            models = _models(train, cal, k)
            test_idx = split.test_idx
            Xtest, ytest = s["F"][test_idx], s["y"][test_idx]
            order_pos = np.arange(len(test_idx))
            valid_test = np.ones(len(test_idx), dtype=bool)
            if k == 0:
                predictions = {"none": raw_model.predict_proba(Xtest)}
            else:
                predictions = {
                    "new_only": models["new_only"].predict_proba(Xtest),
                    "pooled": models["pooled"].predict_proba(Xtest),
                    "pooled_z": models["pooled_z"][0].predict_proba(z_apply(Xtest, models["pooled_z"][1])),
                    "cal_z_only": models["cal_z_only"][0].predict_proba(z_apply(Xtest, models["cal_z_only"][1])),
                }
            own_mask = s["valid"] & np.isin(s["rep"], TRAIN_REPS)
            if k == 0:
                own = None
            else:
                own = None
            for arm, proba in predictions.items():
                m = metrics(proba, raw_model.classes_, ytest, valid_test, order_pos)
                rows.append(dict(session=path.stem, k=k, calibration_seconds=len(split.cal_idx) * WIN_SECONDS,
                                 arm=arm, **{key: value for key, value in m.items() if key != "per_class_recall"},
                                 per_class_recall=repr(m["per_class_recall"])))
        # Reference is recomputed below as a separate k=8 arm.
        own_cal = s["valid"] & np.isin(s["rep"], TRAIN_REPS)
        own_test = s["valid"] & np.isin(s["rep"], TEST_REPS)
        own_model = fit_lda_model(s["F"][own_cal], s["y"][own_cal])
        m = metrics(own_model.predict_proba(s["F"][own_test]), own_model.classes_,
                    s["y"][own_test], np.ones(own_test.sum(), bool), np.arange(own_test.sum()))
        rows.append(dict(session=path.stem, k=8, calibration_seconds=own_cal.sum() * WIN_SECONDS,
                         arm="reference", **{key: value for key, value in m.items() if key != "per_class_recall"},
                         per_class_recall=repr(m["per_class_recall"])))

    print("\nCalibration seconds mean by k:")
    for k in train_seconds:
        print(f"k={k}: {np.mean(train_seconds[k]):.1f}s")
    print("\nMean metrics over 9 sessions:")
    for arm in arms:
        for k in (0, 1, 2, 4, 8):
            selected = [r for r in rows if r["arm"] == arm and int(r["k"]) == k]
            if selected:
                print(arm, k, *(f"{np.mean([float(r[x]) for r in selected]):.4f}" for x in
                                ("accuracy", "balanced_accuracy", "rest_false_pct", "missed_grasp_pct",
                                 "wrong_grasp_pct", "false_events_min")))
    print("\nPer-session metrics: arm/k accuracy balanced rest-false% missed% wrong% false-events/min:")
    for session in sorted({r["session"] for r in rows}):
        values = [r for r in rows if r["session"] == session]
        print(session, " ".join(
            f"{r['arm']}/k{r['k']}={float(r['accuracy']):.3f}/"
            f"{float(r['balanced_accuracy']):.3f}/{float(r['rest_false_pct']):.2f}/"
            f"{float(r['missed_grasp_pct']):.2f}/{float(r['wrong_grasp_pct']):.2f}/"
            f"{float(r['false_events_min']):.3f}"
            for r in values
        ))
    print("\nSessions better than none:")
    for metric, higher in (("accuracy", True), ("false_events_min", False)):
        print(metric, better_than_none(rows, metric, higher))
    print("\nAccuracy vs calibration seconds:")
    for arm in arms:
        for k in (0, 1, 2, 4, 8):
            selected = [r for r in rows if r["arm"] == arm and int(r["k"]) == k]
            if selected:
                print(arm, k, np.mean([float(r["calibration_seconds"]) for r in selected]),
                      np.mean([float(r["accuracy"]) for r in selected]))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/db6_subject")
    parser.add_argument("--out", default="data/db6_subject/calib/calib_results.csv")
    args = parser.parse_args()
    start = time.time()
    run_experiment(args.dir, args.out)
    print(f"Done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
