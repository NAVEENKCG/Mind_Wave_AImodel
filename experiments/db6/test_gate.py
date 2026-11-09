import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import db6_gate as g  # noqa: E402


def test_gate_is_causal_and_negative_control_is_caught():
    F = np.ones((120, 14), dtype=float)
    F[35:55] *= 100.0
    thresholds = g.GateThresholds(theta_on=2.0, theta_off=1.0)
    a = g.controller(F, thresholds)
    F2 = F.copy()
    F2[80:] *= 1000.0
    b = g.controller(F2, thresholds)
    assert np.array_equal(a["command"][:80], b["command"][:80])

    def peeking(x, _):
        out = g.controller(x, thresholds)
        out["command"][:40] = int(np.mean(x[80:, 0]) > 1)
        return out

    assert not np.array_equal(peeking(F, None)["command"][:40], peeking(F2, None)["command"][:40])


def test_hysteresis_requires_on_and_off_runs_without_chatter():
    activity = np.array([0.0, 2.1, 2.1, 1.5, 1.5, 1.5, 0.9, 0.9, 0.9, 0.9, 0.9])
    state = g.hysteresis(activity, theta_on=2.0, theta_off=1.0, k_on=2, k_off=5)
    assert state.tolist() == [False, False, True, True, True, True, True, True, True, True, False]

    no_chatter = g.hysteresis(
        np.array([2.1, 2.1, 1.1, 1.1, 1.1, 2.1]), 2.0, 1.0, 2, 5
    )
    assert no_chatter.tolist() == [False, True, True, True, True, True]


def test_thresholds_only_accept_training_arrays():
    train_F = np.ones((20, 14))
    train_y = np.zeros(20, dtype=int)
    train_valid = np.ones(20, dtype=bool)
    assert g.fit_thresholds(train_F, train_y, train_valid).theta_on >= 0
    with pytest.raises(TypeError):
        g.fit_thresholds(train_F, train_y, train_valid, np.ones((20, 14)))


def test_event_metrics_hand_computed_correct_wrong_missed():
    y = np.array([0, 1, 1, 0, 2, 2, 0, 3, 3, 0])
    rep = np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    pred = np.array([0, 1, 1, 0, 3, 3, 0, 0, 0, 0])
    gate = pred != 0
    result = g.event_metrics(y, rep, pred, gate, np.arange(len(y)), warm=0)
    assert result["commanded"] == 2
    assert result["correct"] == 1
    assert result["wrong"] == 1
    assert result["missed"] == 1
    assert result["early_release"] == 0
    assert np.isnan(result["median_latency"])


def test_segment_units_key_grasp_and_repetition():
    y, rep, seg = [], [], []
    for grasp in range(1, 8):
        for repetition in range(1, 13):
            y.extend([grasp] * 3 + [0] * 2)
            rep.extend([repetition] * 5)
            seg.extend([len(seg) // 5] * 5)
    units = g.session_units(np.array(y), np.array(rep), np.array(seg))
    assert len(units) == 84
    assert len({u["key"] for u in units}) == 84


def test_baseline_metrics_matches_replay_score():
    rng = np.random.default_rng(42)
    n = 80
    classes = np.array([0, 1, 2])
    proba = rng.dirichlet(np.ones(3), size=n)
    y = rng.integers(0, 3, size=n)
    valid = np.ones(n, dtype=bool)
    order = np.arange(n)
    got = g.baseline_metrics(proba, classes, y, valid, order, warm=10)
    import db6_replay as r
    expected = r.score(proba, classes, y, order, valid, 10, [0.0])[0.0]
    assert got["rest_false_pct"] == pytest.approx(100 * expected["rest_false"])
    assert got["false_events_min"] == pytest.approx(expected["events_min"])


def test_summary_from_csv_rows_uses_direction_per_metric():
    rows = [
        {"session": "A", "method": "baseline", "false_events_min": 2, "correct": 5},
        {"session": "A", "method": "gate_raw", "false_events_min": 1, "correct": 6},
        {"session": "B", "method": "baseline", "false_events_min": 2, "correct": 5},
        {"session": "B", "method": "gate_raw", "false_events_min": 3, "correct": 4},
    ]
    assert g.summary_from_rows(rows) == {"false_events_min": 1, "correct": 1}
