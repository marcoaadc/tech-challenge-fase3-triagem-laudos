import csv
from collections import Counter

import pytest

from triage.data.generator import CLASS_WEIGHTS, EXAMS, LABELS, generate_dataset, write_csv


def test_generator_is_deterministic() -> None:
    a = generate_dataset(n_samples=200, seed=1)
    b = generate_dataset(n_samples=200, seed=1)
    assert [r.text for r in a] == [r.text for r in b]
    assert [r.label for r in a] == [r.label for r in b]


def test_generator_changes_with_seed() -> None:
    a = generate_dataset(n_samples=100, seed=1)
    b = generate_dataset(n_samples=100, seed=2)
    assert [r.text for r in a] != [r.text for r in b]


def test_generator_size_labels_and_ids() -> None:
    reports = generate_dataset(n_samples=500, seed=3)
    assert len(reports) == 500
    assert {r.label for r in reports} <= set(LABELS)
    assert {r.exam_type for r in reports} <= set(EXAMS)
    assert len({r.laudo_id for r in reports}) == 500
    assert all(r.text.strip() for r in reports)


def test_class_distribution_follows_weights() -> None:
    reports = generate_dataset(n_samples=6000, seed=42, label_noise=0.0)
    counts = Counter(r.label for r in reports)
    for label, weight in CLASS_WEIGHTS.items():
        assert abs(counts[label] / 6000 - weight) < 0.03


@pytest.mark.parametrize("kwargs", [{"n_samples": 0}, {"label_noise": 1.0}, {"label_noise": -0.1}])
def test_generator_rejects_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        generate_dataset(**{"n_samples": 10, **kwargs})


def test_write_csv_creates_expected_columns(tmp_path) -> None:
    path = write_csv(generate_dataset(n_samples=20, seed=0), tmp_path / "sub" / "laudos.csv")
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 20
    assert set(rows[0]) == {"laudo_id", "exam_type", "text", "label"}
