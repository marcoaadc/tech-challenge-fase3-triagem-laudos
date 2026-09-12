# Contribuindo

## Setup

```bash
poetry install --with train
poetry run pre-commit install
cp .env.example .env
make pipeline        # gera dataset, treina, exporta ONNX e promove o modelo em models/
make test            # 90+ testes
```

## Fluxo de trabalho

1. Crie uma branch a partir de `main` (`feat/...`, `fix/...`, `docs/...`).
2. Rode `make lint test` antes de abrir o PR. O CI executa lint, testes (3.10/3.11), o pipeline de treino, a validação da DAG, o build/smoke test da imagem e a varredura Trivy.
3. Mensagens de commit seguem [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `docs:`, `test:`, `ci:`, `build:`, `perf:`, `chore:`, opcionalmente com escopo (`feat(api): ...`).
4. Mudanças em comportamento do modelo (hiperparâmetros, gerador, gates) devem vir com os artefatos regenerados (`make pipeline`) e com a atualização de `docs/model_card.md`.
5. Registre mudanças relevantes em `CHANGELOG.md` (seção *Não lançado*).

## Convenções

- Python 3.10+, `ruff` (linha de 120 colunas), type hints em funções públicas.
- Configuração só via `Settings` (`pydantic-settings`, prefixo `TRIAGE_`); nada de `os.environ` espalhado.
- Estágios do pipeline são funções `run_*` puras em `src/triage/pipelines/`; a DAG e a CLI apenas as chamam.
- Nenhum artefato binário além de `models/model.onnx` e `models/model.joblib` (limite de 2 MB no pre-commit).
- Segredos nunca no repositório: `.env` está no `.gitignore`; use `TRIAGE_API_KEY` por ambiente.

## Release

```bash
# atualize a versao em pyproject.toml e src/triage/__init__.py, mova o CHANGELOG de "Nao lancado" para a versao
git tag -a vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z      # o workflow release.yml publica ghcr.io/marcoaadc/tech-challenge-fase3-triagem-laudos:X.Y.Z
```
