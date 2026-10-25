import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def load_module(name):
    path = ROOT / "experiments" / "db6" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load_module("db6_baseline")
step3c = load_module("db6_step3c")
drift = load_module("db6_drift_step3")


def test_rest_labels_are_retained_and_leading_rest_is_assigned():
    labels = np.array([0, 0, 2, 2, 0])
    reps = np.array([0, 0, 1, 1, 0])
    objects = np.array([0, 0, 3, 3, 0])

    filled_reps = baseline.ffill_nonzero(reps)
    filled_objects = baseline.ffill_nonzero(objects)
    signal = np.arange(labels.size, dtype=float)[:, None]
    _, y, r, o = baseline.make_windows(signal, labels, filled_reps, filled_objects, fs=10)

    assert np.count_nonzero(y == 0) == 1
    assert r.tolist() == [1, 1]
    assert o.tolist() == [3, 3]


def test_windows_do_not_cross_repetition_or_object_boundaries():
    signal = np.arange(4, dtype=float)[:, None]
    labels = np.ones(4, dtype=int)
    reps = np.array([1, 1, 2, 2])
    objects = np.array([3, 3, 4, 4])

    _, y, r, o = baseline.make_windows(signal, labels, reps, objects, fs=10)

    assert len(y) == 2
    assert r.tolist() == [1, 2]
    assert o.tolist() == [3, 4]

    old_win, old_step = drift.WIN, drift.STEP
    drift.WIN, drift.STEP = 2, 1
    try:
        starts, _, _, keep, _ = drift.window_meta(labels, reps, objects, len(labels))
    finally:
        drift.WIN, drift.STEP = old_win, old_step
    assert starts.tolist() == [0, 1, 2]
    assert keep.tolist() == [True, False, True]


def test_zero_crossing_and_slope_sign_change_features():
    signal = np.array([1.0, -1.0, 1.0], dtype=float)[None, :, None]
    values = baseline.features(signal, np.array([0.0]))[0]

    assert values[2] == 2  # two signal zero crossings
    assert values[3] == 1  # one slope-sign change


def test_feature_threshold_suppresses_small_changes():
    signal = np.array([1.0, -0.01, 1.0], dtype=float)[None, :, None]
    values = step3c.features(signal, np.array([0.1]))[0]

    assert values[2] == 2
    assert values[3] == 1


def test_normalization_parameters_are_fit_only_on_selected_rows():
    train = np.array([[1.0, 10.0], [3.0, 14.0]])
    test = np.array([[100.0, 1000.0]])
    mean, scale = step3c.zfit(train)

    transformed = step3c.zapply(test, mean, scale)

    np.testing.assert_allclose(mean, [2.0, 12.0])
    np.testing.assert_allclose(scale, [1.0, 2.0])
    np.testing.assert_allclose(transformed, [[98.0, 494.0]])


def test_constant_channels_use_finite_normalization():
    mean, scale = step3c.zfit(np.array([[2.0], [2.0]]))
    transformed = step3c.zapply(np.array([[2.0], [3.0]]), mean, scale)

    assert np.isfinite(transformed).all()
    assert scale.tolist() == [step3c.EPS]
