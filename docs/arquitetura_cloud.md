# Decisão Arquitetural — Deploy em Nuvem do Serviço de Triagem

> Entregável da **Etapa 1** do Tech Challenge: análise da estratégia de deploy (batch vs. real-time) e escolha do provedor/serviço de nuvem para o classificador de laudos.

## 1. O cenário

Um hospital de referência recebe laudos e notas clínicas em texto livre durante todo o dia — radiologia, laboratório, ECG, ultrassom e notas de pronto-socorro. O sistema de triagem lê cada texto e devolve uma prioridade (`normal`, `atencao`, `urgente`) para que a equipe médica atenda primeiro os casos com risco de vida.

Dois fatos do domínio orientam a decisão:

1. **A urgência é perecível.** Um laudo de pneumotórax hipertensivo que só é classificado no lote noturno não tem valor; a triagem precisa acontecer no momento em que o laudo é assinado.
2. **A carga é irregular.** O volume acompanha o fluxo do hospital: picos em horários de plantão e após rodadas de exames, vales de madrugada. Existe também o cenário de reprocessamento em massa (ex.: mudar o modelo e reclassificar o histórico).

## 2. Batch vs. real-time

| Critério | Real-time (API síncrona) | Batch (job periódico) |
|---|---|---|
| Latência de decisão | Milissegundos após o laudo ser emitido | Minutos a horas (intervalo do job) |
| Encaixe no fluxo clínico | Integra ao sistema de laudos (RIS/LIS/PEP) via HTTP; a prioridade aparece na fila de atendimento | Só serve para relatórios, auditoria ou reprocessamento |
| Custo por predição | Instância sempre disponível (ou serverless com cold start) | Muito baixo: processa milhares de laudos por execução |
| Complexidade operacional | Health checks, autoscaling, observabilidade contínua | Orquestrador + armazenamento de resultados |
| Tolerância a falhas | Precisa de fallback (fila/retry) para não bloquear o fluxo | Reexecução simples do job |

**Decisão: inferência em tempo real via API REST**, com o batch mantido como caminho secundário (reprocessamento histórico e avaliação offline), reutilizando o mesmo artefato de modelo.

O modelo escolhido (TF-IDF + classificador linear exportado para ONNX) viabiliza essa decisão sem GPU: a inferência unitária fica abaixo de 1 ms em CPU, o que deixa o orçamento de latência quase inteiramente para a rede e o serviço.

## 3. Provedor e serviço: AWS com ECS Fargate

Os três grandes provedores atendem o cenário. A escolha considerou o **encaixe com a arquitetura já construída** (container Docker stateless, métricas Prometheus, pipeline de retreino orquestrado) e o **custo de operação para um serviço leve de CPU**.

| Opção | Vantagens | Por que não foi a escolhida |
|---|---|---|
| **AWS ECS Fargate** (escolhida) | Roda o mesmo container do repositório sem gerenciar servidores; autoscaling por CPU/latência; ALB com health check no `/ready`; integração nativa com CloudWatch e AMP (Prometheus gerenciado) | — |
| AWS Lambda + API Gateway | Custo zero em ociosidade; escala instantânea | Cold start (100–500 ms) inaceitável para p95 clínico; limite de tamanho/execução; menos controle sobre threads do ONNX Runtime |
| AWS SageMaker Endpoint | Deploy de modelo "pronto", A/B nativo | Custo mínimo alto para um modelo de <1 MB; abstrai o container FastAPI/Prometheus que é justamente o objeto do projeto |
| Azure Container Apps | Equivalente ao Fargate, com KEDA e Dapr | Ecossistema do hospital (hipotético) e do time em AWS; menor maturidade de Prometheus gerenciado |
| GCP Cloud Run | Excelente para HTTP stateless, scale-to-zero | Scale-to-zero traz cold start; manter instância mínima aproxima o custo do Fargate sem os demais serviços integrados |

### Arquitetura alvo

```
                         ┌────────────────────────────────────────────────────────────┐
  Sistemas do hospital   │                        AWS (região sa-east-1)               │
  (RIS / LIS / PEP)      │                                                            │
        │  HTTPS         │   ┌──────────┐     ┌───────────────────────────────┐        │
        └───────────────▶│   │   ALB    │────▶│  ECS Fargate (2..N tarefas)   │        │
                         │   │ /ready   │     │  container triage-api         │        │
                         │   └──────────┘     │  FastAPI + ONNX Runtime       │        │
                         │                    │  /predict  /metrics           │        │
                         │                    └──────────────┬────────────────┘        │
                         │                                   │ scrape                 │
                         │   ┌──────────────┐   ┌────────────▼──────────────┐         │
                         │   │  CloudWatch  │   │ Amazon Managed Prometheus │──▶ Grafana (AMG)
                         │   │  logs JSON   │   │   + regras de alerta      │    dashboards
                         │   └──────────────┘   └───────────────────────────┘         │
                         │                                                            │
                         │   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐  │
                         │   │  MWAA        │──▶│  ECR          │──▶│  S3          │  │
                         │   │  (Airflow)   │   │  imagem API   │   │  modelos +   │  │
                         │   │  DAG retreino│   │  imagem treino│   │  datasets    │  │
                         │   └──────────────┘   └───────────────┘   └──────────────┘  │
                         └────────────────────────────────────────────────────────────┘
        GitHub Actions ──▶ lint → test → build → push ECR → deploy ECS (rolling / blue-green)
```

### Mapeamento componente → serviço

| Componente do repositório | Local (Docker Compose) | AWS |
|---|---|---|
| `Dockerfile` (API) | `docker compose up api` | Imagem no **ECR**, tarefa **ECS Fargate** (0,5 vCPU / 1 GB), 2 réplicas mínimas em zonas distintas |
| `/health`, `/ready` | healthcheck do compose | Health check do **ALB** e do ECS; rollout só avança com `/ready` 200 |
| `/metrics` (prometheus_client) | Prometheus local | **Amazon Managed Prometheus** (scrape via ADOT collector sidecar) |
| `monitoring/grafana` | Grafana local | **Amazon Managed Grafana** com o mesmo dashboard JSON |
| Logs JSON | stdout | **CloudWatch Logs** (awslogs driver) com métricas derivadas |
| `airflow/dags` | Airflow standalone | **MWAA** executando a mesma DAG; artefatos em **S3** |
| `models/` | volume local | Bucket **S3** versionado; a tarefa ECS baixa o modelo promovido no start (ou a imagem é rebuildada pelo pipeline) |
| GitHub Actions | `.github/workflows/ci.yml` | Mesmo workflow + job de deploy (`aws-actions/amazon-ecs-deploy-task-definition`) |

## 4. Requisitos não funcionais e como são atendidos

| Requisito | Meta | Mecanismo |
|---|---|---|
| Latência | p95 < 50 ms ponta a ponta na rede do hospital | Modelo ONNX (< 1 ms), 1 worker por vCPU, `intra_op_threads=1`, sem I/O na requisição |
| Disponibilidade | 99,9% | ≥ 2 tarefas em AZs diferentes, ALB com health check, rolling deploy com circuit breaker |
| Escalabilidade | 10× o pico sem intervenção | Target tracking no ECS por CPU (70%) e por `RequestCountPerTarget` |
| Segurança | Dados clínicos em trânsito e em repouso; acesso só por sistemas autorizados | TLS no ALB, VPC privada, IAM por tarefa; **API key** (`X-API-Key`, segredo no Secrets Manager) e **rate limit** na aplicação, com WAF/ALB como primeira camada; sem PHI nos logs |
| Observabilidade | Métricas RED + métricas de modelo | `prometheus_client` → AMP; alertas de erro, latência e distribuição de classes |
| Reprodutibilidade | Modelo promovido rastreável | `metadata.json` com hash SHA-256 dos artefatos e do dataset; `registry.json` com histórico de promoções |
| Retreino | Diário, com quality gate, sem intervenção manual | DAG Airflow: ingest → validate → train → export ONNX → gate → promote → `POST /model/reload` (1 worker por tarefa) ou rolling deploy da imagem |

## 5. Estimativa de custo (ordem de grandeza, us-east-1/sa-east-1)

| Item | Estimativa mensal |
|---|---|
| 2 tarefas Fargate 0,5 vCPU / 1 GB, 24×7 | ~US$ 30–40 |
| ALB | ~US$ 20–25 |
| AMP + AMG (baixo volume de séries) | ~US$ 20–60 |
| MWAA (ambiente pequeno) | ~US$ 300+ (maior custo; alternativa: Airflow em ECS ou EventBridge + ECS RunTask para o retreino diário) |
| ECR + S3 + CloudWatch | < US$ 10 |

O maior custo é o orquestrador gerenciado; para um único retreino diário, executar a DAG como job agendado (EventBridge → ECS RunTask) reduz o total para menos de US$ 100/mês sem perder rastreabilidade.

## 6. Riscos e mitigações

- **Drift de vocabulário** (novos termos, novos exames): monitorar a proporção de classes e a confiança média no Grafana; retreino diário com gate de qualidade.
- **Falso negativo em `urgente`**: o gate exige recall de `urgente` ≥ 0,90; recomenda-se, em produção, tratar `atencao` com baixa confiança como fila de revisão humana.
- **Indisponibilidade da API**: os sistemas clientes devem enfileirar laudos e reenviar (idempotência por `X-Request-ID`); a triagem automática nunca substitui a manual em caso de falha.
- **Dados sensíveis**: a API não persiste o texto dos laudos; logs contêm apenas metadados (rota, status, duração, ID de requisição).
