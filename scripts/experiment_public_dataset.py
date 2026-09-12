"""Experimento de generalizacao: o mesmo pipeline em um corpus medico publico real.

Objetivo: mostrar que o codigo de treino/exportacao nao depende do gerador sintetico. Usa o
*Medical Abstracts TC Corpus* (Schopf et al., 2022): 14.438 resumos medicos em ingles rotulados em
5 condicoes (neoplasms, digestive, nervous, cardiovascular, general pathological). Download
direto do GitHub, sem credenciamento.

Executa os candidatos com os mesmos hiperparametros do projeto, avalia no split de teste
oficial, exporta o melhor para ONNX e mede a paridade. Resultados em
``reports/public_dataset_experiment.{json,md}``.

Por padrao roda Regressao Logistica e Complement NB. O Random Forest e opcional
(``--candidates logreg complement_nb random_forest``): com o vocabulario de bigramas dos resumos
(centenas de milhares de features esparsas) ele leva dezenas de minutos, o que por si so
confirma a decisao do ADR 002 de nao usa-lo em producao.

    poetry run python scripts/experiment_public_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

from triage.nlp.preprocessing import normalize_text
from triage.training.calibration import calibration_report
from triage.training.export import check_parity, export_onnx
from triage.training.pipeline import MODEL_CANDIDATES, build_pipeline, evaluate_pipeline, train_pipeline

logger = logging.getLogger("experiment.public_dataset")

BASE_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/"
FILES = {"train": "medical_tc_train.csv", "test": "medical_tc_test.csv", "labels": "medical_tc_labels.csv"}
CITATION = (
    "Schopf, T., Braun, D., Matthes, F. (2022). Evaluating Unsupervised Text Classification: "
    "Zero-shot and Similarity-based Approaches. NLPIR 2022."
)


def download(name: str, cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / name
    if not target.exists():
        logger.info("baixando %s", name)
        target.write_bytes(urllib.request.urlopen(BASE_URL + name, timeout=60).read())
    return list(csv.DictReader(io.StringIO(target.read_text(encoding="utf-8"))))


def load_corpus(cache_dir: Path) -> tuple[list[str], list[str], list[str], list[str], dict[str, str]]:
    label_names = {r["condition_label"]: r["condition_name"] for r in download(FILES["labels"], cache_dir)}
    train = download(FILES["train"], cache_dir)
    test = download(FILES["test"], cache_dir)
    to_label = lambda r: label_names[r["condition_label"]].strip().lower().replace(" ", "_")  # noqa: E731
    return (
        [r["medical_abstract"] for r in train],
        [to_label(r) for r in train],
        [r["medical_abstract"] for r in test],
        [to_label(r) for r in test],
        label_names,
    )


DEFAULT_CANDIDATES = ("logreg", "complement_nb")


def run(
    cache_dir: Path,
    reports_dir: Path,
    seed: int = 42,
    parity_samples: int = 1000,
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES,
) -> dict:
    x_train, y_train, x_test, y_test, label_names = load_corpus(cache_dir)
    classes = sorted(set(y_train))
    logger.info("corpus: treino=%d teste=%d classes=%s", len(x_train), len(x_test), classes)
    majority = max(classes, key=y_train.count)
    majority_accuracy = y_test.count(majority) / len(y_test)
    logger.info("baseline classe majoritaria (%s): acuracia=%.4f", majority, majority_accuracy)

    results: dict[str, dict] = {}
    pipelines = {}
    for name in candidates:
        pipe = build_pipeline(name, seed=seed)
        start = time.perf_counter()
        train_pipeline(pipe, x_train, y_train)
        fit_seconds = time.perf_counter() - start
        evaluation = evaluate_pipeline(pipe, x_test, y_test, label_list=classes)
        pipelines[name] = pipe
        results[name] = {
            "accuracy": evaluation.accuracy,
            "f1_macro": evaluation.f1_macro,
            "f1_weighted": evaluation.f1_weighted,
            "per_class": evaluation.per_class,
            "fit_seconds": fit_seconds,
            "vocabulary_size": int(len(pipe.named_steps["tfidf"].vocabulary_)),
        }
        logger.info(
            "%-14s acc=%.4f f1_macro=%.4f fit=%.1fs", name, evaluation.accuracy, evaluation.f1_macro, fit_seconds
        )

    best = max(results, key=lambda k: results[k]["f1_macro"])
    best_pipe = pipelines[best]
    onnx_path = export_onnx(best_pipe, cache_dir / "public_model.onnx")
    sample = [normalize_text(t) for t in x_test[:parity_samples]]
    parity = check_parity(best_pipe, onnx_path, sample).as_dict()
    proba = best_pipe.predict_proba([normalize_text(t) for t in x_test])
    calibration = calibration_report(proba, y_test, best_pipe.classes_)
    logger.info("melhor=%s | paridade rotulos=%.4f | ece=%.4f", best, parity["label_agreement"], calibration.ece)

    report = {
        "dataset": {
            "name": "Medical Abstracts TC Corpus",
            "source": "https://github.com/sebischair/Medical-Abstracts-TC-Corpus",
            "citation": CITATION,
            "language": "en",
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "classes": classes,
            "label_names": label_names,
        },
        "seed": seed,
        "majority_baseline": {"label": majority, "accuracy": majority_accuracy},
        "candidates": list(candidates),
        "results": results,
        "best_model": best,
        "onnx_parity": parity,
        "calibration": {k: v for k, v in calibration.as_dict().items() if k != "bins"},
        "onnx_bytes": onnx_path.stat().st_size,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "public_dataset_experiment.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (reports_dir / "public_dataset_experiment.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def render_markdown(report: dict) -> str:
    d = report["dataset"]
    lines = [
        f"# Experimento em dataset publico - {d['name']}",
        "",
        f"Fonte: <{d['source']}> ({d['language']}), treino {d['train_rows']} / teste {d['test_rows']}, "
        f"{len(d['classes'])} classes: {', '.join(d['classes'])}.",
        "",
        "Mesmo pipeline do projeto (TF-IDF 1-2 gramas + candidatos), sem ajuste de hiperparametros.",
        f"Baseline da classe majoritaria (`{report['majority_baseline']['label']}`): "
        f"acuracia {report['majority_baseline']['accuracy']:.4f}.",
        "",
        "| Candidato | Acuracia | F1 macro | F1 ponderado | Vocabulario | Treino (s) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in report["results"].items():
        mark = " (melhor)" if name == report["best_model"] else ""
        lines.append(
            f"| {name}{mark} | {r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['f1_weighted']:.4f} "
            f"| {r['vocabulary_size']} | {r['fit_seconds']:.1f} |"
        )
    p, c = report["onnx_parity"], report["calibration"]
    lines += [
        "",
        f"Paridade scikit-learn x ONNX ({p['n_samples']} amostras): concordancia de rotulos "
        f"{p['label_agreement']:.4f}, desvio maximo de probabilidade {p['max_abs_prob_diff']:.4f}.",
        "",
        f"Calibracao do melhor modelo (teste): Brier {c['brier_score']:.4f}, ECE {c['ece']:.4f}, MCE {c['mce']:.4f}.",
        "",
        f"Citacao: {d['citation']}",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/external/medical_abstracts"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidates", nargs="+", choices=sorted(MODEL_CANDIDATES), default=list(DEFAULT_CANDIDATES))
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    run(args.cache_dir, args.reports_dir, seed=args.seed, candidates=tuple(args.candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
