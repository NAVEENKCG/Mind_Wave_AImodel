import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import db6_calib as c  # noqa: E402


def test_calibration_and_test_masks_are_disjoint_and_test_is_fixed():
    rep = np.array([1, 1, 2, 4, 9, 10, 12, 3])
    valid = np.ones(len(rep), dtype=bool)
    splits = [c.make_split(rep, valid, k) for k in (0, 1, 2, 4)]
    tests = [s.test_idx for s in splits]
    assert all(np.array_equal(tests[0], test) for test in tests[1:])
    assert all(not np.intersect1d(s.cal_idx, s.test_idx).size for s in splits)


def test_z_statistics_depend_only_on_calibration_windows():
    cal = np.arange(20, dtype=float).reshape(5, 4)
    test1 = np.ones((3, 4))
    test2 = np.ones((3, 4)) * 10000
    mu1, sd1 = c.fit_z_stats(cal, test1)
    mu2, sd2 = c.fit_z_stats(cal, test2)
    assert np.array_equal(mu1, mu2)
    assert np.array_equal(sd1, sd2)


def test_predictions_do_not_read_test_labels():
    X = np.arange(30, dtype=float).reshape(10, 3)
    train_y = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2, 1])
    model = c.fit_lda_model(X, train_y)
    p1 = c.predict_arm(model, X)
    p2 = c.predict_arm(model, X, labels=np.arange(10)[::-1])
    assert np.array_equal(p1, p2)

    def mutated_predict(model, X, labels):
        return np.asarray(labels)

    assert not np.array_equal(mutated_predict(model, X, np.arange(10)), p1)


def test_metrics_match_replay_score_on_synthetic_stream():
    rng = np.random.default_rng(4)
    proba = rng.dirichlet(np.ones(3), size=80)
    classes = np.array([0, 1, 2])
    y = rng.integers(0, 3, size=80)
    valid = np.ones(80, dtype=bool)
    order = np.arange(80)
    got = c.metrics(proba, classes, y, valid, order, warm=10)
    import db6_replay as r
    expected = r.score(proba, classes, y, order, valid, 10, [0.0])[0.0]
    assert got["accuracy"] == pytest.approx(expected["acc"])
    assert got["balanced_accuracy"] == pytest.approx(expected["bal"])
    assert got["rest_false_pct"] == pytest.approx(100 * expected["rest_false"])
    assert got["missed_grasp_pct"] == pytest.approx(100 * expected["miss"])
    assert got["false_events_min"] == pytest.approx(expected["events_min"])


def test_segment_table_has_one_row_per_segment():
    y = np.array([1, 1, 0, 2, 2, 0])
    rep = np.array([1, 1, 1, 1, 1, 1])
    seg = np.array([0, 0, 0, 1, 1, 1])
    table = c.segment_table(y, rep, seg)
    assert [(r["grasp"], r["rep"], r["n_windows"]) for r in table] == [(1, 1, 3), (2, 1, 3)]
    assert all(r["fragments"] == 1 for r in table)
