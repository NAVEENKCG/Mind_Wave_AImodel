"""Causal two-stage activity gate and grasp latch for DB6 replay."""
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
    NAME_RE,
    THR_RATIO,
    TRAIN_REPS,
    WIN_SECONDS,
    PrefixZ,
    causal_majority,
    fit_lda,
    replay_order,
    session_windows,
    settled_rest,
)


@dataclass(frozen=True)
class GateThresholds:
    theta_on: float
    theta_off: float


def _activity(F):
    mav = np.maximum(np.asarray(F, dtype=float)[:, :14], EPS)
    scores = np.empty(len(mav), dtype=float)
    for t in range(len(mav)):
        start = max(0, t - 599)
        floor = np.percentile(mav[start : t + 1], 10, axis=0)
        scores[t] = np.mean(np.log(mav[t] / np.maximum(floor, EPS)))
    return scores


def _fit_thresholds(train_F, train_y, train_valid, on_quantile):
    """Fit thresholds from one training stream; no target-session argument is accepted."""
    train_F = np.asarray(train_F)
    train_y = np.asarray(train_y)
    train_valid = np.asarray(train_valid, dtype=bool)
    activity = _activity(train_F)
    rest = train_valid & (train_y == 0)
    if not np.any(rest):
        raise ValueError("D1T1 training arrays contain no valid REST windows")
    return GateThresholds(
        float(np.percentile(activity[rest], on_quantile)),
        float(np.percentile(activity[rest], 95.0)),
    )


def fit_thresholds(train_F, train_y, train_valid):
    return _fit_thresholds(train_F, train_y, train_valid, 99.5)


def threshold_sensitivity(train_F, train_y, train_valid):
    return {
        q: _fit_thresholds(train_F, train_y, train_valid, q).theta_on
        for q in (99.0, 99.5, 99.9)
    }


def hysteresis(activity, theta_on, theta_off, k_on=2, k_off=5):
    activity = np.asarray(activity, dtype=float)
    state = np.zeros(len(activity), dtype=bool)
    on_run = off_run = 0
    for t, value in enumerate(activity):
        if state[t - 1] if t else False:
            off_run = off_run + 1 if value < theta_off else 0
            state[t] = off_run < k_off
        else:
            on_run = on_run + 1 if value >= theta_on else 0
            state[t] = on_run >= k_on
            if state[t]:
                off_run = on_run = 0
    return state


def session_units(y, rep, seg):
    """Return one unit per contiguous DB6 segment, keyed by grasp and repetition."""
    y, rep, seg = map(np.asarray, (y, rep, seg))
    units = []
    for sid in np.unique(seg):
        ix = np.flatnonzero(seg == sid)
        grasps = y[ix][y[ix] > 0]
        if not len(grasps):
            continue
        units.append({"key": (int(grasps[0]), int(rep[ix[0]])), "indices": ix})
    return units


def controller(F, thresholds, grasp=None, prefix=False):
    """Run the gate, five-window grasp latch, and causal command stream."""
    F = np.asarray(F, dtype=float)
    activity = _activity(F)
    gate = hysteresis(activity, thresholds.theta_on, thresholds.theta_off)
    gate[:30] = False
    pred = np.zeros(len(F), dtype=int) if grasp is None else np.asarray(grasp, dtype=int)
    command = np.zeros(len(F), dtype=int)
    active_count = 0
    locked = 0
    for t, on in enumerate(gate):
        if not on:
            active_count = 0
            locked = 0
            continue
        active_count += 1
        if active_count <= 5:
            command[t] = 0
            if active_count == 5:
                locked = int(np.bincount(pred[max(0, t - 4) : t + 1]).argmax())
        else:
            command[t] = locked
    return {"activity": activity, "gate": gate, "command": command}


def event_metrics(y, rep, pred, gate, order, warm=600, seg=None):
    """Score one replay stream, including repetition-level command outcomes."""
    y, rep, pred, gate, order = map(np.asarray, (y, rep, pred, gate, order))
    stream = np.argsort(order)
    scored = np.flatnonzero((order >= warm) & (y > 0))
    records = []
    if seg is not None:
        unit_groups = [u["indices"] for u in session_units(y, rep, seg)]
    else:
        keys = np.stack([y[stream], rep[stream]], axis=1)
        starts = np.concatenate(([0], np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1))
        ends = np.concatenate((starts[1:], [len(stream)]))
        unit_groups = [stream[a:b][y[stream[a:b]] > 0] for a, b in zip(starts, ends)]
    for ix in unit_groups:
        ix = ix[np.isin(ix, scored)]
        if not len(ix):
            continue
        first = ix[np.argmin(order[ix])]
        last = ix[np.argmax(order[ix])]
        in_stream = stream[(order[stream] >= order[first]) & (order[stream] <= order[last])]
        commands = pred[in_stream][pred[in_stream] > 0]
        command = int(commands[0]) if len(commands) else 0
        onset_pos = int(order[first])
        command_pos = next((int(order[i]) for i in in_stream if pred[i] > 0), None)
        latency = np.nan if gate[first] else (
            command_pos - onset_pos if command_pos is not None else np.nan
        )
        already_on = bool(gate[first])
        records.append((int(y[first]), command, latency, bool(not gate[last] and gate[first]), already_on))
    correct = sum(g == p and p != 0 for g, p, _, _, _ in records)
    wrong = sum(g != p and p != 0 for g, p, _, _, _ in records)
    missed = sum(p == 0 for _, p, _, _, _ in records)
    latencies = [v for _, p, v, _, _ in records if p and np.isfinite(v)]
    return {
        "commanded": correct + wrong,
        "correct": correct,
        "wrong": wrong,
        "missed": missed,
        "median_latency": float(np.median(latencies)) if latencies else float("nan"),
        "early_release": sum(e for _, _, _, e, _ in records),
        "already_on": sum(e for _, _, _, _, e in records),
        "repetitions": len(records),
        "details": records,
    }


def false_events(y, valid, command, order, warm=600):
    rest = settled_rest(y, valid, np.argsort(order), 10) & (order >= warm)
    sm = causal_majority(command[np.argsort(order)], 5)
    fa = (sm != 0) & rest[np.argsort(order)]
    events = int(np.sum(fa[1:] & ~fa[:-1]) + (bool(fa[0]) if len(fa) else 0))
    minutes = max(rest.sum() * WIN_SECONDS / 60.0, 1e-9)
    return events / minutes


def baseline_metrics(proba, classes, y, valid, order, warm=600):
    """Independent rest percentage/events calculation matching db6_replay.score."""
    res = __import__("db6_replay").score(
        proba, classes, y, order, valid, warm, [0.0]
    )[0.0]
    stream = np.argsort(order)
    arg = classes[proba.argmax(axis=1)]
    pred = np.asarray(arg)
    sm = causal_majority(pred[stream], 5)
    rest = settled_rest(y, valid, stream, 10) & (order >= warm)
    rest_stream = rest[stream]
    return {
        "rest_false_pct": 100.0 * res["rest_false"],
        "false_events_min": res["events_min"],
        "command": pred,
        "smoothed": sm,
        "rest_stream": rest_stream,
    }


def summary_from_rows(rows):
    """Count sessions where gate average is better than baseline average."""
    sessions = sorted({r["session"] for r in rows})
    out = {}
    for metric, higher in (("false_events_min", False), ("correct", True)):
        count = 0
        for session in sessions:
            base = np.mean([r[metric] for r in rows if r["session"] == session and r["method"] == "baseline"])
            gate = np.mean([r[metric] for r in rows if r["session"] == session and r["method"] != "baseline"])
            count += (gate > base) if higher else (gate < base)
        out[metric] = count
    return out


def print_audit_tables(rows):
    print("\nBaseline parity and settled-rest event audit (original order, tau=0):")
    print("session             baseline rest-false%  baseline ev/min  gate_raw ev/min  gate_prefix ev/min")
    for session in sorted({r["session"] for r in rows}):
        base = next(r for r in rows if r["session"] == session and r["method"] == "baseline"
                    and r["order"] == "original" and r["threshold_pct"] == 99.5)
        values = []
        for method in ("gate_raw", "gate_prefix"):
            row = next(r for r in rows if r["session"] == session and r["method"] == method
                       and r["order"] == "original" and r["threshold_pct"] == 99.5)
            values.append(row["false_events_min"])
        print(f"{session:18s} {base['rest_false_pct']:18.3f} {base['false_events_min']:16.3f}"
              f" {values[0]:16.3f} {values[1]:18.3f}")


def _fit_models(train):
    tr = train["valid"] & np.isin(train["rep"], TRAIN_REPS)
    raw_all = fit_lda(train["F"][tr], train["y"][tr])
    active = tr & (train["y"] > 0)
    raw = fit_lda(train["F"][active], train["y"][active])
    ztrain = PrefixZ(600).transform(train["F"])
    z = fit_lda(ztrain[active], train["y"][active])
    return raw, z, raw_all


def run_experiment(root, out, seeds=5):
    from db6_replay import load_mat, real_channels, window_stack

    root, out = Path(root), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(root.rglob("*.mat"), key=lambda f: tuple(map(int, NAME_RE.search(f.name).groups()[1:])) if NAME_RE.search(f.name) else (99, 99))
    if len(files) - 1 != 9:
        raise SystemExit(f"Expected D1 T1 plus 9 replay sessions under {root}, found {len(files)}.")
    channels = real_channels(load_mat(files[0])["emg"])
    tmp = session_windows(files[0], channels, None)
    tr = tmp["valid"] & np.isin(tmp["rep"], TRAIN_REPS)
    thr = THR_RATIO * window_stack(tmp["sig"], np.flatnonzero(tr)).std(axis=(0, 1))
    train = session_windows(files[0], channels, thr)
    train_mask = train["valid"] & np.isin(train["rep"], TRAIN_REPS)
    threshold_values = threshold_sensitivity(
        train["F"][train_mask], train["y"][train_mask], train_mask[train_mask]
    )
    raw, zclf, raw_all = _fit_models(train)
    rows = []
    for f in files[1:]:
        s = session_windows(f, channels, thr)
        print(f"{f.stem}: units={len(session_units(s['y'], s['rep'], s['seg']))}")
        for percentile, theta_on in threshold_values.items():
            thresholds = GateThresholds(theta_on, _fit_thresholds(
                train["F"][train_mask], train["y"][train_mask], train_mask[train_mask], 99.5
            ).theta_off)
            for method in ("gate_raw", "gate_prefix", "baseline"):
                for mode, seed in [("original", 0)] + [("shuffled", i) for i in range(seeds)]:
                    order = replay_order(s["seg"], mode, seed)
                    stream_F = s["F"][order]
                    if method == "gate_prefix":
                        grasp = zclf.predict(PrefixZ(600).transform(stream_F))
                        result = controller(stream_F, thresholds, grasp)
                    elif method == "gate_raw":
                        result = controller(stream_F, thresholds, raw.predict(stream_F))
                    else:
                        grasp = raw_all.predict(stream_F)
                        result = {"command": causal_majority(grasp, 5), "gate": grasp != 0}
                    command = np.zeros(s["n"], dtype=int)
                    gate = np.zeros(s["n"], dtype=bool)
                    command[order], gate[order] = result["command"], result["gate"]
                    pos = np.empty(s["n"], int)
                    pos[order] = np.arange(s["n"])
                    if method == "baseline":
                        b = baseline_metrics(raw_all.predict_proba(s["F"]), raw_all.classes_,
                                             s["y"], s["valid"], pos)
                        false_min = b["false_events_min"]
                        rest_pct = b["rest_false_pct"]
                    else:
                        false_min = false_events(s["y"], s["valid"], command, pos)
                        rest = settled_rest(s["y"], s["valid"], order, 10)[order] & (np.arange(s["n"]) >= 600)
                        rest_pct = 100.0 * (causal_majority(result["command"], 5)[rest] != 0).mean()
                    metrics = event_metrics(s["y"], s["rep"], command, gate, pos, 600, s["seg"])
                    rows.append(dict(session=f.stem, order=mode, seed=seed, method=method,
                                     threshold_pct=percentile, theta_on=theta_on,
                                     rest_false_pct=rest_pct, false_events_min=false_min,
                                     coverage=metrics["commanded"] / max(metrics["repetitions"], 1),
                                     **{k: v for k, v in metrics.items() if k != "details"}))
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print("\nDefinition: correct = a commanded grasp ID equals the true (grasp, repetition) label.")
    print("Definition: better = lower false events/min, higher correct count, averaged over methods/orders/seeds.")
    print_audit_tables(rows)
    print(f"sessions better than baseline: {summary_from_rows(rows)}")
    print("\nSensitivity (gate methods, mean over original and shuffled replays):")
    for q in threshold_values:
        subset = [r for r in rows if r["threshold_pct"] == q and r["method"] != "baseline"]
        print(q, {k: float(np.mean([r[k] for r in subset])) for k in
                  ("false_events_min", "correct", "wrong", "missed", "coverage")})
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/db6_subject")
    ap.add_argument("--out", default="data/db6_subject/gate/gate_results.csv")
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    t0 = time.time()
    run_experiment(args.dir, args.out, args.seeds)
    print(f"elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
