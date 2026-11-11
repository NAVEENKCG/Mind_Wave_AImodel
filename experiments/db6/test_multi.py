import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import db6_multi as m  # noqa: E402


def test_subjects_are_isolated():
    a = np.array([1.0, 2.0])
    b = np.array([100.0, 200.0])
    stats_a = (a.mean(), a.std())
    stats_b = (b.mean(), b.std())
    assert stats_a != stats_b
    assert not np.array_equal(np.concatenate([a]), np.concatenate([b]))


def test_training_session_belongs_to_subject():
    files = [Path("S3_D1_T1.mat"), Path("S3_D1_T2.mat")]
    m.check_subject_training(files, 3)
    with pytest.raises(AssertionError):
        m.check_subject_training(files, 2)


def test_completeness_flags_missing_grasp_or_repetition():
    training = {"y": np.array([1] * 10 + [2] * 10),
                "rep": np.array([1] * 10 + [1] * 10),
                "valid": np.ones(20, dtype=bool)}
    complete, missing = m.completeness([1, 2], training)
    assert not complete
    assert (1, 2, 0) in missing
    assert (2, 1, 10) not in missing


def test_training_subject_mismatch_is_caught():
    with pytest.raises(AssertionError):
        m.check_subject_training([Path("S2_D1_T1.mat")], 3)


def test_mutation_copy_training_subject_b_on_subject_a_data_is_caught():
    with pytest.raises(AssertionError):
        m.check_subject_training([Path("S2_D1_T1.mat")], 3)
