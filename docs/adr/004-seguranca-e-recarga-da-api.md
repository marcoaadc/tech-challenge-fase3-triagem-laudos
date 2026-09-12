# ADR 004 — Segurança da API e recarga do modelo em runtime

**Status:** aceito · **Data:** 2026-09-12

## Contexto

A versão 1.0 expunha `/predict` sem autenticação nem limite de taxa, e só passava a servir um modelo novo após reiniciar o container. Para laudos médicos, qualquer integração real exige controle de acesso; e o ciclo retreino → produção da DAG ficava dependente de intervenção manual.

## Decisões

1. **API key por header (`X-API-Key`)**, configurada por `TRIAGE_API_KEY` e comparada em tempo constante (`hmac.compare_digest`). Aplicada a `/predict`, `/predict/batch` e `/model/reload`. Vazia = desativada, para manter o Compose local e o CI sem segredos. Rotas operacionais (`/health`, `/ready`, `/metrics`, `/model/info`) ficam abertas: são consumidas por orquestrador e Prometheus, não por clientes.
   - Alternativas: OAuth2/JWT (excesso para integração servidor-a-servidor dentro da rede do hospital; fica para o gateway/ALB em nuvem) e mTLS (adequado, mas operado pela infraestrutura, não pela aplicação).
2. **Limite de taxa por cliente** com janela deslizante em memória (`TRIAGE_RATE_LIMIT_PER_MINUTE`), resposta `429` com `Retry-After` e métrica `triage_rate_limited_total`. Chave = primeiro IP de `X-Forwarded-For` ou IP do socket. Limitação assumida: estado por processo; com várias réplicas o limite é por réplica. Em nuvem, o rate limit do ALB/WAF é a fonte de verdade e este é a segunda camada.
3. **`POST /model/reload`** troca o predictor em memória a partir de `models_dir`. A carga é feita por completo (incluindo warm-up) antes da troca: se falhar, o modelo atual continua servindo e a resposta é `503`. A última task da DAG chama esse endpoint, fechando o ciclo sem restart.
   - Alternativa considerada: *file watcher* recarregando automaticamente ao detectar mudança em `models/`. Rejeitada: a DAG copia três arquivos e um watcher poderia carregar um estado intermediário; a chamada explícita acontece depois da promoção completa.
4. **Múltiplos workers** via `TRIAGE_WORKERS` no entrypoint; com mais de um processo, o `prometheus_client` entra em modo multiprocesso (`PROMETHEUS_MULTIPROC_DIR`) e `/metrics` agrega todos os workers. Consequência: o reload precisa ser chamado uma vez por worker em cenários multi-processo (o endpoint só atualiza o processo que atendeu). Documentado; para produção multi-worker, o caminho recomendado é rolling deploy da imagem com o novo modelo, não reload.

## Consequências

- (+) Integração segura por padrão quando configurada; sem impacto no fluxo local.
- (+) Retreino noturno chega à API sem intervenção, com verificação de versão na própria DAG.
- (−) Rate limit em memória não é compartilhado entre réplicas.
- (−) Reload em modo multi-worker atualiza apenas um processo; usar 1 worker por container (padrão) ou rolling deploy.
