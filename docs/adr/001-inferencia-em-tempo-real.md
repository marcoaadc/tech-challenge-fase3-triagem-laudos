# ADR 001 — Inferência em tempo real via API REST

**Status:** aceito · **Data:** 2026-09-10

## Contexto

O hospital precisa priorizar laudos assim que são emitidos. A latência de decisão faz parte do valor clínico: um laudo urgente classificado horas depois não muda a conduta.

## Decisão

Servir o classificador como uma **API REST síncrona** (FastAPI em container Docker), com contrato `POST /predict` (um laudo) e `POST /predict/batch` (lote pequeno), health checks separados de *liveness* (`/health`) e *readiness* (`/ready`), e métricas Prometheus em `/metrics`.

O processamento em lote continua possível reutilizando o mesmo artefato (`models/model.onnx`) em jobs, mas não é o caminho primário.

## Consequências

- (+) A prioridade aparece no sistema clínico no momento da assinatura do laudo.
- (+) Contrato simples de integrar (HTTP/JSON) e de testar (TestClient, smoke test em CI).
- (−) Exige disponibilidade contínua, autoscaling e observabilidade; mitigado com ECS Fargate + ALB e alertas.
- (−) O modelo precisa caber no orçamento de latência: escolha de modelo leve e exportação para ONNX (ADR 002 e 003).
