"""Estagio 4 - promocao: quality gate e publicacao do candidato no diretorio servido pela API.

O treino escreve em ``models/candidate``; so depois de passar nos criterios abaixo o candidato
e copiado para ``models/`` (o "slot de producao" lido pela API). Um historico e mantido em
``models/registry.json``. Se o gate reprovar, o estagio falha e o modelo em producao permanece.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from triage.config.settings import get_settings
from triage.pipelines._cli import build_parser, read_json, setup_logging, write_json
from triage.training.metadata import METADATA_FILENAME, ONNX_FILENAME, SKLEARN_FILENAME, ModelMetadata

logger = logging.getLogger("triage.pipelines.promote")

REGISTRY_FILENAME = "registry.json"


class PromotionGateError(RuntimeError):
    """O candidato nao atende aos criterios minimos de qualidade."""


@dataclass(frozen=True)
class Gates:
    min_f1_macro: float = 0.90
    min_urgent_recall: float = 0.90
    min_onnx_label_agreement: float = 0.99
    max_onnx_prob_diff: float = 0.05


def check_gates(metadata: ModelMetadata, gates: Gates | None = None) -> list[str]:
    """Retorna a lista de violacoes (vazia = aprovado)."""
    gates = gates or Gates()
    violations: list[str] = []
    test = metadata.metrics.get("test", {})
    parity = metadata.metrics.get("onnx_parity", {})

    f1 = float(test.get("f1_macro", 0.0))
    if f1 < gates.min_f1_macro:
        violations.append(f"f1_macro(teste)={f1:.4f} < {gates.min_f1_macro}")
    urgent_recall = float(test.get("urgent_recall", 0.0))
    if urgent_recall < gates.min_urgent_recall:
        violations.append(f"urgent_recall(teste)={urgent_recall:.4f} < {gates.min_urgent_recall}")

    if "onnx" not in metadata.artifacts:
        violations.append("artefato ONNX ausente (execute o estagio export)")
    else:
        agreement = float(parity.get("label_agreement", 0.0))
        if agreement < gates.min_onnx_label_agreement:
            violations.append(f"paridade ONNX rotulos={agreement:.4f} < {gates.min_onnx_label_agreement}")
        prob_diff = float(parity.get("max_abs_prob_diff", 1.0))
        if prob_diff > gates.max_onnx_prob_diff:
            violations.append(f"paridade ONNX prob diff={prob_diff:.4f} > {gates.max_onnx_prob_diff}")
    return violations


def run_promote(
    candidate_dir: Path, target_dir: Path, gates: Gates | None = None, reports_dir: Path | None = None
) -> dict:
    gates = gates or Gates()
    candidate_dir, target_dir = Path(candidate_dir), Path(target_dir)
    metadata = ModelMetadata.load(candidate_dir)
    violations = check_gates(metadata, gates)
    decision = {
        "model_version": metadata.model_version,
        "model_type": metadata.model_type,
        "promoted": not violations,
        "violations": violations,
        "gates": gates.__dict__,
        "decided_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    if reports_dir is not None:
        write_json(Path(reports_dir) / "promotion.json", decision)
    if violations:
        logger.error("candidato %s REPROVADO: %s", metadata.model_version, "; ".join(violations))
        raise PromotionGateError("; ".join(violations))

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in (SKLEARN_FILENAME, ONNX_FILENAME, METADATA_FILENAME):
        shutil.copy2(candidate_dir / name, target_dir / name)

    registry_path = target_dir / REGISTRY_FILENAME
    registry = read_json(registry_path) if registry_path.exists() else {"current": None, "history": []}
    registry["history"].append(
        {
            "model_version": metadata.model_version,
            "model_type": metadata.model_type,
            "promoted_at": decision["decided_at"],
            "f1_macro_test": metadata.metrics["test"]["f1_macro"],
            "urgent_recall_test": metadata.metrics["test"]["urgent_recall"],
            "onnx_sha256": metadata.artifacts["onnx"]["sha256"],
        }
    )
    registry["current"] = metadata.model_version
    write_json(registry_path, registry)
    logger.info("candidato %s PROMOVIDO para %s", metadata.model_version, target_dir)
    return decision


def main(argv: list[str] | None = None) -> None:
    settings = get_settings()
    parser = build_parser("Aplica o quality gate e promove o candidato para o diretorio servido pela API.")
    parser.add_argument("--candidate-dir", type=Path, default=settings.models_dir / "candidate")
    parser.add_argument("--target-dir", type=Path, default=settings.models_dir)
    parser.add_argument("--min-f1-macro", type=float, default=Gates.min_f1_macro)
    parser.add_argument("--min-urgent-recall", type=float, default=Gates.min_urgent_recall)
    parser.add_argument("--reports-dir", type=Path, default=settings.reports_dir)
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    gates = Gates(min_f1_macro=args.min_f1_macro, min_urgent_recall=args.min_urgent_recall)
    run_promote(args.candidate_dir, args.target_dir, gates, args.reports_dir)


if __name__ == "__main__":
    main()
