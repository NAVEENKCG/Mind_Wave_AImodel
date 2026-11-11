import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import db6_cgate as c  # noqa: E402
import db6_gate  # noqa: E402


def test_thresholds_use_calibration_rest_only():
    features = np.ones((8, 8))
    features[4:] *= 10
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    a = c.fit_gate_thresholds(features, labels, np.ones(8, bool))
    changed = features.copy()
    changed[4:] *= 10000
    b = c.fit_gate_thresholds(changed, labels, np.ones(8, bool))
    assert np.array_equal(a["rest_median"], b["rest_median"])
    assert a["theta_on"] == pytest.approx(b["theta_on"])


def test_activity_and_gate_do_not_read_future_rows():
    features = np.ones((80, 8))
    features[20:40] *= 5
    thresholds = {"rest_median": np.ones(2), "theta_on": 1.0, "theta_off": 0.5}
    a_scores, a_gate = c.gate_state(features, thresholds)
    changed = features.copy()
    changed[60:] *= 1000
    b_scores, b_gate = c.gate_state(changed, thresholds)
    assert np.array_equal(a_scores[:60], b_scores[:60])
    assert np.array_equal(a_gate[:60], b_gate[:60])


def test_hysteresis_timing():
    activity = np.array([0.0, 2.0, 2.0, 0.9, 0.9, 0.9, 0.9, 0.9])
    features = np.ones((len(activity), 8))
    features[:, :2] = np.exp(activity)[:, None]
    _, state = c.gate_state(
        features, {"rest_median": np.ones(2), "theta_on": 1.0, "theta_off": 1.0}
    )
    assert state.tolist() == [False, False, True, True, True, True, True, False]


def test_commit_abstains_on_split_vote():
    predictions = np.array([1, 2, 1, 2, 3, 0, 0])
    gate = np.ones(len(predictions), dtype=bool)
    command = c.commit_predictions(predictions, gate, 5, minimum_share=0.6)
    assert np.all(command == 0)


def test_commit_locks_until_gate_off():
    predictions = np.array([1, 1, 1, 1, 1, 2, 2, 2, 0])
    gate = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0], dtype=bool)
    command = c.commit_predictions(predictions, gate, 5)
    assert command[4:8].tolist() == [1, 1, 1, 1]
    assert command[8] == 0
