import pytest

from triage.pipelines.train import split_dataset
from triage.training.pipeline import (
    MODEL_CANDIDATES,
    EvaluationResult,
    build_pipeline,
    evaluate_pipeline,
    select_best,
    train_pipeline,
)


def test_build_pipeline_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="model_type desconhecido"):
        build_pipeline("transformer")


@pytest.mark.parametrize("model_type", sorted(MODEL_CANDIDATES))
def test_candidates_can_be_built(model_type: str) -> None:
    pipe = build_pipeline(model_type, seed=0)
    assert [name for name, _ in pipe.steps] == ["tfidf", "clf"]


def test_split_is_stratified_and_disjoint(dataset_df) -> None:
    train, val, test = split_dataset(dataset_df, seed=1)
    assert len(train) + len(val) + len(test) == len(dataset_df)
    assert abs(len(test) / len(dataset_df) - 0.15) < 0.01
    assert set(train["laudo_id"]).isdisjoint(test["laudo_id"])
    for label in ("normal", "atencao", "urgente"):
        full = (dataset_df["label"] == label).mean()
        assert abs((test["label"] == label).mean() - full) < 0.03


def test_train_and_evaluate_reaches_quality_floor(dataset_df) -> None:
    train, _, test = split_dataset(dataset_df, seed=1)
    pipe = train_pipeline(build_pipeline("logreg", seed=1), train["text"], train["label"])
    result = evaluate_pipeline(pipe, test["text"], test["label"])
    assert result.f1_macro > 0.85
    assert result.urgent_recall > 0.85
    assert sum(sum(row) for row in result.confusion_matrix) == len(test)
    flat = result.flat_metrics("val_")
    assert "val_urgente_recall" in flat and "val_f1_macro" in flat


def _result(f1: float, urgent_recall: float) -> EvaluationResult:
    labels = ("normal", "atencao", "urgente")
    per_class = {lb: {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 10} for lb in labels}
    per_class["urgente"]["recall"] = urgent_recall
    return EvaluationResult(accuracy=0.9, f1_macro=f1, f1_weighted=0.9, per_class=per_class, confusion_matrix=[])


def test_select_best_prefers_f1_then_urgent_recall() -> None:
    assert select_best({"a": _result(0.90, 0.9), "b": _result(0.95, 0.8)}) == "b"
    assert select_best({"a": _result(0.95, 0.9), "b": _result(0.95, 0.8)}) == "a"
    with pytest.raises(ValueError):
        select_best({})
