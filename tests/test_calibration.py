import numpy as np
import pytest

from triage.training.calibration import calibration_report, render_reliability_table


def test_perfectly_calibrated_one_hot_predictions() -> None:
    classes = ["a", "b", "c"]
    y = ["a", "b", "c", "a"]
    proba = np.eye(3)[[0, 1, 2, 0]]
    report = calibration_report(proba, y, classes, n_bins=5)
    assert report.brier_score == 0.0
    assert report.ece == 0.0
    assert report.accuracy == 1.0
    assert report.n_samples == 4
    assert len(report.bins) == 1 and report.bins[0].count == 4


def test_overconfident_predictions_have_high_ece() -> None:
    classes = ["a", "b"]
    # Sempre 90% de confianca em "a", mas so metade dos rotulos e "a".
    proba = np.tile([0.9, 0.1], (10, 1))
    y = ["a"] * 5 + ["b"] * 5
    report = calibration_report(proba, y, classes, n_bins=10)
    assert report.accuracy == 0.5
    assert report.ece == pytest.approx(0.4)
    assert report.mce == pytest.approx(0.4)
    assert report.brier_score == pytest.approx(0.5 * 0.02 + 0.5 * 1.62)


def test_reliability_bins_are_ordered_and_cover_samples() -> None:
    rng = np.random.default_rng(0)
    raw = rng.random((200, 3))
    proba = raw / raw.sum(axis=1, keepdims=True)
    classes = ["x", "y", "z"]
    y = [classes[i] for i in proba.argmax(axis=1)]
    report = calibration_report(proba, y, classes, n_bins=10)
    assert sum(b.count for b in report.bins) == 200
    assert [b.lower for b in report.bins] == sorted(b.lower for b in report.bins)
    table = render_reliability_table(report)
    assert table.startswith("| Faixa de confianca") and table.count("\n") == len(report.bins) + 1


def test_invalid_shapes_raise() -> None:
    with pytest.raises(ValueError):
        calibration_report(np.ones((3, 2)), ["a", "b", "a"], ["a", "b", "c"])
    with pytest.raises(ValueError):
        calibration_report(np.ones((3, 2)) / 2, ["a", "b"], ["a", "b"])
