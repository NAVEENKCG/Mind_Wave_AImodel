"""Independent held-out-subject runner for NinaPro DB6."""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db6_calib as calib  # noqa: E402
import db6_replay as replay  # noqa: E402


SUBJECTS = (2, 3, 4)
CAL_KS = (0, 1, 2, 4)
REPLAY_ARMS = ("none", "prefix", "roll")
CAL_ARMS = ("new_only", "pooled", "pooled_z", "cal_z_only")
ALL_ARMS = ("none", "prefix", "roll", "new_only", "pooled",
            "pooled_z", "cal_z_only", "reference")
REQUIRED_REPS = (1, 2, 3, 4, 9, 10, 11, 12)


def session_files(root):
    def key(path):
        match = replay.NAME_RE.search(path.name)
        if not match:
            return (99, 99, str(path))
        return (int(match.group(2)), int(match.group(3)), str(path))

    return sorted(root.rglob("*.mat"), key=key)


def subject_number(path):
    match = replay.NAME_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot identify subject in {path}")
    return int(match.group(1))


def check_subject_training(files, expected_subject):
    if not files:
        raise ValueError(f"No .mat files found for subject {expected_subject}")
    match = replay.NAME_RE.search(files[0].name)
    if not match or (int(match.group(1)), int(match.group(2)),
                     int(match.group(3))) != (expected_subject, 1, 1):
        raise AssertionError(
            f"Subject {expected_subject} training file must be D1 T1; got {files[0].name}"
        )


def _counts(session):
    counts = defaultdict(int)
    for label, rep, valid in zip(session["y"], session["rep"], session["valid"]):
        if valid and label > 0:
            counts[(int(label), int(rep))] += 1
    return dict(counts)


def completeness(training_labels, session):
    counts = _counts(session)
    missing = []
    for grasp in training_labels:
        for rep in REQUIRED_REPS:
            count = counts.get((grasp, rep), 0)
            if count < 10:
                missing.append((grasp, rep, count))
    return (not missing), missing


def session_report(path, session, training_labels):
    reps = defaultdict(list)
    for grasp in sorted(training_labels):
        reps[grasp] = sorted({int(r) for y, r in zip(session["y"], session["rep"])
                              if y == grasp})
    complete, missing = completeness(training_labels, session)
    return {
        "path": str(path),
        "size_mb": path.stat().st_size / 1e6,
        "labels": sorted({int(y) for y in session["y"] if y > 0}),
        "repetitions": dict(reps),
        "valid_windows": int(session["valid"].sum()),
        "complete": complete,
        "missing": missing,
    }


def _threshold(training):
    mask = training["valid"] & np.isin(training["rep"], replay.TRAIN_REPS)
    if not mask.any():
        raise ValueError("Training session has no valid repetitions 1-8")
    raw_windows = replay.window_stack(training["sig"], np.flatnonzero(mask))
    return replay.THR_RATIO * raw_windows.std(axis=(0, 1))


def _metrics_row(proba, classes, session, order_pos, warm, arm, k, order, seed):
    result = replay.score(proba, classes, session["y"], order_pos,
                          session["valid"], warm, [0.0])[0.0]
    predicted = classes[np.asarray(proba).argmax(axis=1)]
    scored_grasp = session["valid"] & (order_pos >= warm) & (session["y"] > 0)
    wrong = scored_grasp & (predicted > 0) & (predicted != session["y"])
    return {
        "kind": "replay", "order": order, "seed": seed, "arm": arm, "k": k,
        "accuracy": result["acc"], "balanced_accuracy": result["bal"],
        "rest_false_pct": 100 * result["rest_false"],
        "wrong_grasp_pct": 100 * wrong.sum() / max(scored_grasp.sum(), 1),
        "false_events_min": result["events_min"],
        "calibration_seconds": 0.0, "n": result["n"],
    }


def replay_rows(session, train, clf_raw, clf_z, warmup, halflife, seeds):
    rows = []
    orders = [("original", 0)] + [("shuffled", seed) for seed in range(seeds)]
    for order_name, seed in orders:
        order = replay.replay_order(session["seg"], order_name, seed)
        position = np.empty(session["n"], dtype=int)
        position[order] = np.arange(session["n"])
        stream_features = session["F"][order]
        for corrector in replay.make_correctors(warmup, halflife):
            if corrector.name == "offline":
                continue
            transformed = corrector.transform(stream_features)
            classifier = clf_raw if corrector.name == "none" else clf_z
            stream_proba = classifier.predict_proba(transformed)
            proba = np.empty_like(stream_proba)
            proba[order] = stream_proba
            rows.append(_metrics_row(
                proba, classifier.classes_, session, position, warmup,
                corrector.name, 0, order_name, seed,
            ))
    return rows


def calibration_rows(session, train, warmup=600):
    train_mask = train["valid"] & np.isin(train["rep"], replay.TRAIN_REPS)
    rows = []
    test_mask = session["valid"] & np.isin(session["rep"], replay.TEST_REPS)
    if not test_mask.any():
        return rows
    test_features = session["F"][test_mask]
    test_labels = session["y"][test_mask]
    for k in CAL_KS:
        split = calib.make_split(session["rep"], session["valid"], k)
        if k == 0:
            continue
        models = calib._models(train, session, k)
        for arm in CAL_ARMS:
            model = models[arm]
            stats = None
            if isinstance(model, tuple):
                model, stats = model
                features = calib.z_apply(test_features, stats)
            else:
                features = test_features
            proba = model.predict_proba(features)
            result = calib.metrics(
                proba, model.classes_, test_labels,
                np.ones(len(test_labels), dtype=bool),
                np.arange(len(test_labels)), warm=0,
            )
            wrong = result["wrong_grasp_pct"]
            rows.append({
                "kind": "calibration", "order": "original", "seed": 0,
                "arm": arm, "k": k, "accuracy": result["accuracy"],
                "balanced_accuracy": result["balanced_accuracy"],
                "rest_false_pct": result["rest_false_pct"],
                "wrong_grasp_pct": wrong,
                "false_events_min": result["false_events_min"],
                "calibration_seconds": len(split.cal_idx) * replay.WIN_SECONDS,
                "n": result["n"],
            })
    own_train = session["valid"] & np.isin(session["rep"], replay.TRAIN_REPS)
    own_test = session["valid"] & np.isin(session["rep"], replay.TEST_REPS)
    if own_test.any():
        own_model = replay.fit_lda(session["F"][own_train], session["y"][own_train])
        proba = own_model.predict_proba(session["F"][own_test])
        result = calib.metrics(
            proba, own_model.classes_, session["y"][own_test],
            np.ones(int(own_test.sum()), dtype=bool),
            np.arange(int(own_test.sum())), warm=0,
        )
        rows.append({
            "kind": "calibration", "order": "original", "seed": 0,
            "arm": "reference", "k": 8, "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "rest_false_pct": result["rest_false_pct"],
            "wrong_grasp_pct": result["wrong_grasp_pct"],
            "false_events_min": result["false_events_min"],
            "calibration_seconds": int(own_train.sum()) * replay.WIN_SECONDS,
            "n": result["n"],
        })
    return rows


def run_subject(subject_root, output_root, subject_id, subject1_channels,
                warmup=600, halflife=300, seeds=3):
    files = session_files(subject_root)
    check_subject_training(files, subject_id)
    training_path = files[0]
    training_data = replay.load_mat(training_path)
    channels = replay.real_channels(training_data["emg"])
    print(f"Subject {subject_id} channels ({len(channels)}): {channels}")
    if subject1_channels is not None and channels != subject1_channels:
        print(f"CHANNEL DIFFERENCE from subject 1: {channels} vs {subject1_channels}")
    raw_training = replay.session_windows(training_path, channels, None)
    threshold = _threshold(raw_training)
    training = replay.session_windows(training_path, channels, threshold)
    training_labels = sorted({int(y) for y in training["y"] if y > 0})
    train_complete, train_missing = completeness(training_labels, training)
    print(f"Training session: {training_path.name}; COMPLETE={train_complete}; missing={train_missing}")
    if not train_complete:
        return {"subject": subject_id, "skipped": True, "reason": "incomplete D1T1"}
    train_mask = training["valid"] & np.isin(training["rep"], replay.TRAIN_REPS)
    clf_raw = replay.fit_lda(training["F"][train_mask], training["y"][train_mask])
    train_mu, train_sd = training["F"][train_mask].mean(0), training["F"][train_mask].std(0) + replay.EPS
    clf_z = replay.fit_lda((training["F"][train_mask] - train_mu) / train_sd,
                           training["y"][train_mask])
    subject_rows = []
    complete_count = incomplete_count = 0
    for path in files:
        try:
            session = replay.session_windows(path, channels, threshold)
            report = session_report(path, session, training_labels)
            print(f"{path.name}: size={report['size_mb']:.2f} MB labels={report['labels']} "
                  f"reps={report['repetitions']} valid={report['valid_windows']} "
                  f"{'COMPLETE' if report['complete'] else 'INCOMPLETE'}")
            print(f"  missing={report['missing']}")
            complete_count += int(report["complete"])
            incomplete_count += int(not report["complete"])
            if path == training_path:
                continue
            new_rows = replay_rows(
                session, training, clf_raw, clf_z, warmup, halflife, seeds
            )
            new_rows.extend(calibration_rows(session, training, warmup))
            for row in new_rows:
                row["session"] = path.name
                row["session_complete"] = report["complete"]
            subject_rows.extend(new_rows)
        except Exception as exc:
            print(f"FAILURE subject {subject_id} session {path}: {type(exc).__name__}: {exc}")
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / "multi_results.csv"
    fields = ["kind", "session", "session_complete", "order", "seed", "arm", "k", "accuracy",
              "balanced_accuracy", "rest_false_pct", "wrong_grasp_pct",
              "false_events_min", "calibration_seconds", "n"]
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(subject_rows)
    return {"subject": subject_id, "skipped": False, "path": str(out),
            "complete": complete_count, "incomplete": incomplete_count,
            "rows": subject_rows}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/db6_subjects")
    parser.add_argument("--out", default="results/multi")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)
    subject1 = Path("data/db6_subject")
    subject1_files = session_files(subject1)
    subject1_channels = None
    if subject1_files:
        subject1_channels = replay.real_channels(
            replay.load_mat(subject1_files[0])["emg"]
        )
    summaries = []
    for subject_id in SUBJECTS:
        try:
            summaries.append(run_subject(
                Path(args.root) / f"s{subject_id}",
                Path(args.out) / f"s{subject_id}",
                subject_id, subject1_channels, seeds=args.seeds,
            ))
        except Exception as exc:
            print(f"FAILURE subject {subject_id}: {type(exc).__name__}: {exc}")
            summaries.append({"subject": subject_id, "skipped": True, "reason": str(exc)})
    print("\nSUMMARY (original-order rows; means exclude incomplete sessions)")
    print("metric\t" + "\t".join(f"S{s}" for s in SUBJECTS))
    metrics = (
        ("baseline_accuracy", "none", 0, "accuracy"),
        ("reference_accuracy", "reference", 8, "accuracy"),
        ("prefix_accuracy", "prefix", 0, "accuracy"),
        ("roll_accuracy", "roll", 0, "accuracy"),
        ("new_only_k4_accuracy", "new_only", 4, "accuracy"),
        ("pooled_k4_accuracy", "pooled", 4, "accuracy"),
        ("pooled_z_k4_accuracy", "pooled_z", 4, "accuracy"),
        ("cal_z_only_k4_accuracy", "cal_z_only", 4, "accuracy"),
        ("calibration_seconds_k4", "new_only", 4, "calibration_seconds"),
        ("rest_false_pct_none", "none", 0, "rest_false_pct"),
        ("wrong_grasp_pct_none", "none", 0, "wrong_grasp_pct"),
        ("false_events_min_none", "none", 0, "false_events_min"),
    )
    values = {}
    for summary in summaries:
        if summary.get("skipped"):
            values[summary["subject"]] = {label: "SKIPPED" for label, *_ in metrics}
            continue
        complete_rows = [r for r in summary["rows"] if r.get("session_complete")]

        def mean(arm, k=None, field="accuracy"):
            values = [float(r[field]) for r in complete_rows
                      if r["order"] == "original" and r["arm"] == arm
                      and (k is None or int(r["k"]) == k)]
            return float(np.mean(values)) if values else float("nan")
        values[summary["subject"]] = {
            label: mean(arm, k, field) for label, arm, k, field in metrics
        }
        values[summary["subject"]]["complete_sessions"] = summary["complete"]
        values[summary["subject"]]["incomplete_sessions"] = summary["incomplete"]
    for label, *_ in metrics:
        print(label + "\t" + "\t".join(
            str(values.get(subject, {}).get(label, "MISSING")) for subject in SUBJECTS
        ))
    for label in ("complete_sessions", "incomplete_sessions"):
        print(label + "\t" + "\t".join(
            str(values.get(subject, {}).get(label, "MISSING")) for subject in SUBJECTS
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
