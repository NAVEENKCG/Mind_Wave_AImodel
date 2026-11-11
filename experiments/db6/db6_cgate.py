"""Subject-1 calibrated gate and commit audit."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import db6_calib as calib  # noqa: E402
import db6_replay as replay  # noqa: E402
from db6_gate import hysteresis  # noqa: E402


CAL_KS = (2, 4)
COMMIT_NS = (5, 10, 15)


def activity_scores(features, rest_median):
    features = np.asarray(features, dtype=float)
    channels = features.shape[1] // 4
    mav = np.maximum(features[:, :channels], replay.EPS)
    median = np.maximum(np.asarray(rest_median, dtype=float), replay.EPS)
    return np.mean(np.log(mav / median), axis=1)


def fit_gate_thresholds(cal_features, cal_labels, cal_valid):
    features = np.asarray(cal_features, dtype=float)
    labels = np.asarray(cal_labels)
    valid = np.asarray(cal_valid, dtype=bool)
    channels = features.shape[1] // 4
    rest = valid & (labels == 0)
    if not rest.any():
        raise ValueError("Calibration data contains no valid REST windows")
    rest_median = np.median(features[rest, :channels], axis=0)
    scores = activity_scores(features[rest], rest_median)
    return {
        "rest_median": rest_median,
        "theta_on": float(np.percentile(scores, 99.0)),
        "theta_off": float(np.percentile(scores, 95.0)),
        "theta_on_99_5": float(np.percentile(scores, 99.5)),
    }


def gate_state(features, thresholds):
    scores = activity_scores(features, thresholds["rest_median"])
    state = hysteresis(
        scores, thresholds["theta_on"], thresholds["theta_off"],
        k_on=2, k_off=5,
    )
    return scores, state


def majority_predictions(predictions, width=5):
    return replay.causal_majority(np.asarray(predictions, dtype=int), width)


def commit_predictions(predictions, gate, n, minimum_share=0.6):
    predictions = np.asarray(predictions, dtype=int)
    gate = np.asarray(gate, dtype=bool)
    if n <= 0:
        raise ValueError("Commit window count must be positive")
    command = np.zeros(len(predictions), dtype=int)
    committed = False
    active = []
    for i, on in enumerate(gate):
        if not on:
            committed = False
            active = []
            continue
        if not committed:
            active.append(int(predictions[i]))
            if len(active) == n:
                values, counts = np.unique(active, return_counts=True)
                winner = int(values[counts.argmax()])
                share = float(counts.max() / n)
                if winner > 0 and share >= minimum_share:
                    committed = True
                    command[i] = winner
            continue
        command[i] = winner
    return command


def merged_units(y, rep, seg):
    y, rep, seg = map(np.asarray, (y, rep, seg))
    units = []
    for sid in np.unique(seg):
        indices = np.flatnonzero(seg == sid)
        active = y[indices][y[indices] > 0]
        if not len(active):
            continue
        key = (int(active[0]), int(rep[indices[0]]))
        if units and units[-1]["key"] == key and units[-1]["indices"][-1] + 1 == indices[0]:
            units[-1]["indices"] = np.concatenate((units[-1]["indices"], indices))
        else:
            units.append({"key": key, "indices": indices.copy()})
    return units


def score_commands(y, rep, valid, command, gate, order, warm=0, seg=None):
    y, rep, valid, command, gate, order = map(
        np.asarray, (y, rep, valid, command, gate, order)
    )
    stream = np.argsort(order)
    scored = (order >= warm) & valid
    settled = replay.settled_rest(y, valid, stream, 10) & (order >= warm)
    smoothed = replay.causal_majority(command[stream], 5)
    rest = settled[stream] & (smoothed != 0)
    rest_windows = max(int(settled[stream].sum()), 1)
    false_events = int(np.sum(rest[1:] & ~rest[:-1]) + (bool(rest[0]) if len(rest) else 0))
    units = merged_units(y, rep, np.arange(len(y)) if seg is None else seg)
    records = []
    for unit in units:
        indices = unit["indices"]
        indices = indices[scored[indices] & (y[indices] > 0)]
        if not len(indices):
            continue
        stream_indices = indices[np.argsort(order[indices])]
        nonzero = command[stream_indices][command[stream_indices] > 0]
        predicted = int(nonzero[0]) if len(nonzero) else 0
        grasp = unit["key"][0]
        active_gate = bool(gate[stream_indices].any())
        abstain = int(predicted == 0 and active_gate)
        first = int(stream_indices[0])
        last = int(stream_indices[-1])
        command_indices = stream_indices[command[stream_indices] > 0]
        latency = float("nan")
        if len(command_indices) and not gate[first]:
            latency = int(order[command_indices[0]] - order[first])
        records.append({
            "grasp": grasp, "prediction": predicted,
            "correct": int(predicted == grasp),
            "wrong": int(predicted > 0 and predicted != grasp),
            "missed": int(predicted == 0 and not abstain),
            "abstain": abstain,
            "latency": latency,
            "early_release": int(gate[first] and not gate[last]),
            "already_on": int(gate[first]),
        })
    latencies = [r["latency"] for r in records if np.isfinite(r["latency"])]
    return {
        "rest_false_pct": 100.0 * rest.sum() / rest_windows,
        "false_events_min": false_events / max(rest_windows * replay.WIN_SECONDS / 60.0, 1e-9),
        "correct": sum(r["correct"] for r in records),
        "wrong": sum(r["wrong"] for r in records),
        "missed": sum(r["missed"] for r in records),
        "abstain": sum(r["abstain"] for r in records),
        "median_latency": float(np.median(latencies)) if latencies else float("nan"),
        "early_release": sum(r["early_release"] for r in records),
        "already_on": sum(r["already_on"] for r in records),
        "units": len(records),
    }


def session_files(root):
    return calib._session_files(root)


def run_subject1(root, output):
    files, channels, threshold, training = calib._prepare(root)
    train_mask = training["valid"] & np.isin(training["rep"], replay.TRAIN_REPS)
    train_features = training["F"][train_mask]
    train_labels = training["y"][train_mask]
    rows = []
    for path in files[1:]:
        session = replay.session_windows(path, channels, threshold)
        for k in CAL_KS:
            split = calib.make_split(session["rep"], session["valid"], k)
            cal_mask = session["valid"] & np.isin(session["rep"], np.arange(1, k + 1))
            cal_features = session["F"][cal_mask]
            cal_labels = session["y"][cal_mask]
            gate_thresholds = fit_gate_thresholds(cal_features, cal_labels, np.ones(len(cal_labels), bool))
            test_idx = split.test_idx
            test_features = session["F"][test_idx]
            test_labels = session["y"][test_idx]
            test_valid = np.ones(len(test_idx), dtype=bool)
            order = np.arange(len(test_idx))
            model = calib.fit_lda_model(
                cal_features[cal_labels > 0], cal_labels[cal_labels > 0]
            )
            predicted = model.predict(test_features)
            majority = majority_predictions(predicted, 5)
            scores, gate = gate_state(test_features, gate_thresholds)
            for arm, command in (("classifier_only", majority), ("gate", majority * gate)):
                metric = score_commands(
                    test_labels, session["rep"][test_idx], test_valid,
                    command, gate if arm == "gate" else np.zeros(len(command), bool),
                    order, seg=session["seg"][test_idx],
                )
                rows.append(_row(path, k, arm, 0, gate_thresholds, metric))
            for n in COMMIT_NS:
                command = commit_predictions(predicted, gate, n)
                metric = score_commands(test_labels, session["rep"][test_idx],
                                        test_valid, command, gate, order,
                                        seg=session["seg"][test_idx])
                rows.append(_row(path, k, "gate_commit", n, gate_thresholds, metric))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        fields = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output}")
    return rows


def _row(path, k, arm, commit_n, thresholds, metric):
    return {
        "session": path.name, "k": k, "arm": arm, "commit_n": commit_n,
        "theta_on": thresholds["theta_on"],
        "theta_off": thresholds["theta_off"],
        "theta_on_99_5": thresholds["theta_on_99_5"],
        **metric,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/db6_subject")
    parser.add_argument("--out", default="results/cgate/s1/cgate_results.csv")
    args = parser.parse_args(argv)
    run_subject1(Path(args.dir), Path(args.out))


if __name__ == "__main__":
    main()
