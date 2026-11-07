import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import db6_cued_calibration as c  # noqa: E402


def blocks():
    result = []
    for grasp in (1, 2):
        for rep in range(1, 6):
            y = np.array([0] * 4 + [grasp] * 6 + [0] * 5)
            result.append(dict(
                grasp=grasp, rep=rep, F=np.full((15, 2), rep + grasp, float),
                y=y, valid=np.ones(15, bool), duration_seconds=1.0,
                original_start=(grasp * 100 + rep) * 2.0,
                original_stop=(grasp * 100 + rep) * 2.0 + 1.0,
            ))
    return result


def test_split_uses_common_actual_labels():
    maps = [{1: [1, 2, 3, 4, 5], 2: [1, 2, 3, 4, 5]},
            {1: [1, 2, 3, 4], 2: [1, 2, 3, 4, 5]}]
    old = c.repetition_map
    c.repetition_map = lambda _: maps.pop(0)
    try:
        pool, heldout, counts = c.choose_split([Path("a"), Path("b")], 2)
    finally:
        c.repetition_map = old
    assert pool == {1: [1, 2], 2: [1, 2]}
    assert heldout == {1: [3, 4], 2: [3, 4, 5]}
    assert counts == {1: 4, 2: 5}


def test_calibration_and_heldout_blocks_are_disjoint_and_heldout_is_common():
    b = blocks()
    pool = {1: [1, 2, 3, 4], 2: [1, 2, 3, 4]}
    heldout = {1: [5], 2: [5]}
    expected = {(1, 5), (2, 5)}
    for k in (0, 1, 2, 4):
        cal = {(x["grasp"], x["rep"]) for x in c.select_blocks(b, {g: v[:k] for g, v in pool.items()})}
        ev = {(x["grasp"], x["rep"]) for x in c.select_blocks(b, heldout)}
        assert not cal & ev
        assert ev == expected


def test_frozen_fit_does_not_depend_on_evaluation_data_and_leaky_control_fails():
    rng = np.random.default_rng(1)
    train_F = rng.normal(size=(80, 4))
    train_y = np.tile([0, 1], 40)
    cal_F = rng.normal(size=(20, 4))
    cal_y = np.tile([0, 1], 10)
    fitted1 = c.fit_conditions(train_F, train_y, cal_F, cal_y)
    evaluation = rng.normal(size=(30, 4))
    evaluation2 = evaluation.copy()
    evaluation2[-1] += 100
    fitted2 = c.fit_conditions(train_F, train_y, cal_F, cal_y)
    assert np.array_equal(fitted1["cued-stats"][1].mean, fitted2["cued-stats"][1].mean)
    assert np.array_equal(fitted1["cued-refit"][0].coef_, fitted2["cued-refit"][0].coef_)

    def leaky_stats(calibration, heldout):
        return c.FrozenStats(np.vstack((calibration, heldout)).mean(0),
                             np.vstack((calibration, heldout)).std(0) + c.EPS)
    assert not np.array_equal(leaky_stats(cal_F, evaluation).mean,
                              leaky_stats(cal_F, evaluation2).mean)


def test_cued_stats_and_refit_at_zero_match_raw():
    rng = np.random.default_rng(2)
    F = rng.normal(size=(100, 4))
    y = np.tile([0, 1], 50)
    methods = c.fit_conditions(F, y, np.empty((0, 4)), np.empty(0, dtype=int))
    assert np.array_equal(methods["raw"][1].transform(F), methods["cued-stats"][1].transform(F))
    assert np.array_equal(methods["raw"][0].predict(F), methods["cued-refit"][0].predict(F))


def test_calibration_windows_are_not_scored():
    b = blocks()
    ev = c.select_blocks(b, {1: [5], 2: [5]})
    F = np.vstack([x["F"] for x in ev])
    y = np.concatenate([x["y"] for x in ev])
    fitted = c.fit_conditions(F, y, np.empty((0, 2)), np.empty(0, dtype=int))
    scored = c.evaluate(ev, fitted, [0.0])["raw"][0.0]
    assert scored["n"] == sum(x["valid"].sum() for x in ev)


def test_block_scoring_resets_majority_and_has_no_join_window():
    b = blocks()[:2]
    clf = type("Classifier", (), {"classes_": np.array([0, 1])})()
    proba = np.tile([[0.99, 0.01]], (15, 1))
    result = c._score_blocks([(b[0], proba, clf.classes_), (b[1], proba, clf.classes_)], [0.0])[0.0]
    assert result["n"] == 30
    assert all(len(x["F"]) == 15 for x in b)


def _predictions(block, fitted):
    clf, corrector = fitted["cued-refit"]
    proba = clf.predict_proba(corrector.transform(block["F"]))
    pred = clf.classes_[proba.argmax(axis=1)]
    return pred, c.causal_majority(pred)


def test_replay_outputs_up_to_t_ignore_later_evaluation_samples():
    rng = np.random.default_rng(3)
    train_F = rng.normal(size=(100, 3))
    train_y = np.tile([0, 1], 50)
    cal_F = rng.normal(size=(10, 3))
    cal_y = np.tile([0, 1], 5)
    fitted = c.fit_conditions(train_F, train_y, cal_F, cal_y)
    base = blocks()[0].copy()
    base["F"] = rng.normal(size=(15, 3))
    changed = dict(base, F=base["F"].copy())
    changed["F"][10:] += 100
    a_raw, a_smoothed = _predictions(base, fitted)
    b_raw, b_smoothed = _predictions(changed, fitted)
    assert np.array_equal(a_raw[:10], b_raw[:10])
    assert np.array_equal(a_smoothed[:10], b_smoothed[:10])
