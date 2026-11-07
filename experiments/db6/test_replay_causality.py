"""
Tests for db6_replay.py. Run from the repo root:  python -m pytest experiments/db6 -q

The central test: changing LATER input must never change EARLIER decisions.
It is checked at three levels (features, correctors, end-to-end predictions), and each
check is paired with a NEGATIVE CONTROL (a deliberately non-causal component that the
same check must catch). A test that cannot fail is not evidence.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import db6_replay as r  # noqa: E402


def synth_raw(n_seconds=40, fs=2000, n_ch=6, seed=0):
    rng = np.random.default_rng(seed)
    n = int(n_seconds * fs)
    gain = 1.0 + 3.0 * (np.sin(np.linspace(0, 6 * np.pi, n))[:, None] > 0)     # alternating rest/active
    return rng.normal(size=(n, n_ch)) * 1e-5 * gain


@pytest.fixture(scope="module")
def raw():
    return synth_raw()


@pytest.fixture(scope="module")
def thr(raw):
    return 0.02 * r.decimate_slice(r.bandpass_causal(raw, r.FS_RAW), r.FS_RAW).std(axis=0)


# ------------------------------------------------------------- level 1: features
def test_features_of_early_windows_ignore_later_samples(raw, thr):
    T = raw.shape[0] // 2
    rng = np.random.default_rng(1)
    raw2 = raw.copy()
    raw2[T:] = rng.normal(size=raw2[T:].shape) * 1e-3                 # wildly different future
    F1, F2 = r.stream_features(raw, thr), r.stream_features(raw2, thr)
    done = r.window_end_raw(len(F1)) <= T                              # windows fully received before T
    assert done.sum() > 10
    assert np.array_equal(F1[done], F2[done])
    assert not np.array_equal(F1[~done], F2[~done])                    # control: later windows DO change


def test_zero_phase_filtfilt_would_leak_negative_control(raw):
    """Negative control: a zero-phase filter changes earlier output when later input changes,
    which is exactly what the feature test above must be able to detect."""
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, r.BAND, btype="bandpass", fs=r.FS_RAW, output="sos")
    T = raw.shape[0] // 2
    raw2 = raw.copy()
    raw2[T:] *= 7.0
    a = sosfiltfilt(sos, raw, axis=0)[: T - 2000]
    b = sosfiltfilt(sos, raw2, axis=0)[: T - 2000]
    assert not np.array_equal(a, b)


# ------------------------------------------------------------ level 2: correctors
@pytest.mark.parametrize("make", [lambda: r.Identity(), lambda: r.PrefixZ(40), lambda: r.RollZ(30)])
def test_causal_correctors_do_not_peek(make):
    rng = np.random.default_rng(2)
    F = rng.normal(size=(300, 12)) + np.linspace(0, 3, 300)[:, None]
    for t in (10, 39, 40, 41, 150, 298):
        assert r.is_causal(make(), F, t, rng), f"{make().name} peeked past row {t}"


def test_offline_corrector_is_caught_as_non_causal():
    rng = np.random.default_rng(3)
    F = rng.normal(size=(300, 12))
    assert not r.is_causal(r.OfflineZ(), F, 100, rng)


def test_leaky_corrector_is_caught():
    class Leaky:
        name = "leaky"

        def transform(self, F):
            return F - F[-1]                                          # uses the LAST row for every row

    rng = np.random.default_rng(4)
    assert not r.is_causal(Leaky(), rng.normal(size=(100, 5)), 50, rng)


def test_prefix_is_identity_during_warmup_then_standardises():
    rng = np.random.default_rng(5)
    F = rng.normal(loc=5.0, scale=2.0, size=(400, 4))
    out = r.PrefixZ(100).transform(F)
    assert np.array_equal(out[:100], F[:100])
    head = F[:100]
    expected = (F[100:] - head.mean(0)) / (head.std(0) + r.EPS)
    assert np.allclose(out[100:], expected)


def test_roll_tracks_a_level_shift_but_prefix_does_not():
    rng = np.random.default_rng(6)
    F = np.vstack([rng.normal(0, 1, (400, 3)), rng.normal(8, 1, (400, 3))])
    roll = r.RollZ(50).transform(F)[-100:].mean()
    prefix = r.PrefixZ(100).transform(F)[-100:].mean()
    assert abs(roll) < 1.0 and prefix > 5.0


# ---------------------------------------------------- level 3: end-to-end decisions
def _pipeline(raw_signal, thr, corrector, clf):
    F = r.stream_features(raw_signal, thr)
    return clf.predict(corrector.transform(F)), len(F)


@pytest.fixture(scope="module")
def clf(raw, thr):
    F = r.stream_features(raw, thr)
    y = (np.arange(len(F)) // 20 % 2).astype(int)
    return r.fit_lda((F - F.mean(0)) / (F.std(0) + r.EPS), y)


@pytest.mark.parametrize("make", [lambda: r.Identity(), lambda: r.PrefixZ(60), lambda: r.RollZ(40)])
def test_end_to_end_decisions_do_not_depend_on_the_future(raw, thr, clf, make):
    T = int(raw.shape[0] * 0.6)
    rng = np.random.default_rng(7)
    raw2 = raw.copy()
    raw2[T:] = rng.normal(size=raw2[T:].shape) * 5e-4
    p1, n = _pipeline(raw, thr, make(), clf)
    p2, _ = _pipeline(raw2, thr, make(), clf)
    done = r.window_end_raw(n) <= T
    assert done.sum() > 50
    assert np.array_equal(p1[done], p2[done])


def test_end_to_end_catches_the_offline_corrector(raw, thr, clf):
    T = int(raw.shape[0] * 0.6)
    raw2 = raw.copy()
    raw2[T:] = np.random.default_rng(8).normal(size=raw2[T:].shape) * 5e-3
    p1, n = _pipeline(raw, thr, r.OfflineZ(), clf)
    p2, _ = _pipeline(raw2, thr, r.OfflineZ(), clf)
    done = r.window_end_raw(n) <= T
    assert not np.array_equal(p1[done], p2[done])


# ----------------------------------------------------- replay order integrity
def make_seg():
    # 3 grasps x 4 reps, each repetition = 30 grasp windows + 10 rest windows
    key = []
    for g in (1, 2, 3):
        for rep in range(1, 5):
            key += [(g, rep)] * 40
    key = np.array([a * 1000 + b for a, b in key])
    return np.concatenate([[0], np.cumsum(key[1:] != key[:-1])])


def test_shuffle_keeps_whole_repetitions_contiguous():
    seg = make_seg()
    order = r.replay_order(seg, "shuffled", seed=3)
    assert sorted(order) == list(range(len(seg)))                     # a permutation, nothing lost
    s = seg[order]
    runs = np.flatnonzero(np.diff(s) != 0)
    assert len(runs) + 1 == seg.max() + 1                             # each segment is ONE contiguous run
    for sid in range(seg.max() + 1):
        w = order[s == sid]
        assert np.array_equal(w, np.sort(w))                          # window order inside a segment unchanged


def test_shuffle_is_deterministic_and_actually_shuffles():
    seg = make_seg()
    a, b = r.replay_order(seg, "shuffled", 1), r.replay_order(seg, "shuffled", 1)
    c = r.replay_order(seg, "shuffled", 2)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert not np.array_equal(a, np.arange(len(seg)))
    assert np.array_equal(r.replay_order(seg, "original"), np.arange(len(seg)))


def test_original_order_prefix_is_one_grasp_but_shuffled_prefix_mixes_grasps():
    """Why the shuffle exists: in file order the first windows come from a single grasp."""
    seg = make_seg()
    grasp_of_window = np.repeat([1, 2, 3], 160)
    first = slice(0, 100)
    assert len(set(grasp_of_window[r.replay_order(seg, "original")][first])) == 1
    mixes = [len(set(grasp_of_window[r.replay_order(seg, "shuffled", s)][first])) for s in range(10)]
    assert max(mixes) > 1


# ----------------------------------------------------------- label bookkeeping
def test_ffill_nonzero_gives_rest_the_repetition_it_follows():
    v = np.array([0, 0, 1, 1, 0, 0, 2, 2, 0, 3, 0])
    assert r.ffill_nonzero(v).tolist() == [1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3]


def test_replay_order_rejects_unknown_mode():
    with pytest.raises(ValueError):
        r.replay_order(make_seg(), "backwards")


# --------------------------------------------------------------- scoring rules
def test_all_methods_are_scored_on_identical_windows_and_rejection_defaults_to_rest():
    n = 200
    rng = np.random.default_rng(9)
    proba = rng.dirichlet(np.ones(3), size=n)
    classes = np.array([0, 1, 2])
    y = rng.integers(0, 3, size=n)
    valid = np.ones(n, bool)
    pos = np.arange(n)
    res = r.score(proba, classes, y, pos, valid, warm=50, taus=[0.0, 1.01])
    assert res[0.0]["n"] == res[1.01]["n"] == 150
    # tau above any confidence rejects everything -> never a false activation, but every grasp is missed
    assert res[1.01]["rest_false"] == 0.0 and res[1.01]["miss"] == 1.0


def _one_hot(pred, n_classes=2):
    p = np.zeros((len(pred), n_classes))
    p[np.arange(len(pred)), pred] = 1.0
    return p


def test_false_events_ignore_release_lag_but_count_real_spurious_activations():
    y = np.array([0] * 100 + [1] * 60 + [0] * 100)
    valid = np.ones(len(y), bool)
    pos = np.arange(len(y))
    classes = np.array([0, 1])
    clean = r.score(_one_hot(y), classes, y, pos, valid, warm=0, taus=[0.0], guard=10)[0.0]
    assert clean["events_min"] == 0.0                       # perfect decoding: the smoother's release lag is not a false event
    spurious = y.copy()
    spurious[230:238] = 1                                   # 8 windows of grasp during settled rest
    bad = r.score(_one_hot(spurious), classes, y, pos, valid, warm=0, taus=[0.0], guard=10)[0.0]
    assert bad["events_min"] > 0.0
    # the 8-window burst is ONE event (a run), not eight
    one_event = 1.0 / (r.settled_rest(y, valid, np.arange(len(y)), 10).sum() * r.WIN_SECONDS / 60.0)
    assert abs(bad["events_min"] - one_event) < 1e-9


def test_settled_rest_excludes_the_guard_after_a_grasp():
    y = np.array([0] * 30 + [1] * 10 + [0] * 30)
    valid = np.ones(len(y), bool)
    s = r.settled_rest(y, valid, np.arange(len(y)), guard=10)
    assert not s[:10].any() and s[10:30].all()              # start-of-stream guard, then settled
    assert not s[30:50].any() and s[50:].all()              # the grasp and the 10 windows after it are excluded
