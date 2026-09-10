# Model Card — Classificador de Urgência de Laudos (`tfidf+logreg`)

## Detalhes do modelo

| Campo | Valor |
|---|---|
| Tarefa | Classificação de texto em 3 classes de urgência: `normal`, `atencao`, `urgente` |
| Tipo | TF-IDF (1-2 gramas, `sublinear_tf`, `min_df=2`) + Regressão Logística (`C=5`, `class_weight="balanced"`) |
| Framework | scikit-learn 1.7 (treino) · ONNX Runtime 1.23 (inferência) |
| Vocabulário | 4.955 n-gramas |
| Artefatos | `models/model.joblib` (146 KB), `models/model.onnx` (225 KB), `models/metadata.json` (métricas, hashes SHA-256, hiperparâmetros) |
| Versão | `20260910.213610-440596bb` (timestamp UTC + hash do dataset), registrada em `models/registry.json` |
| Código | `src/triage/training/pipeline.py` (treino), `src/triage/training/export.py` (ONNX), `src/triage/serving/predictor.py` (inferência) |
| Idioma | Português (Brasil) |

### Seleção do modelo

Três candidatos foram treinados sobre o mesmo split (70/15/15, estratificado, `seed=42`) e comparados por **F1 macro na validação**, com desempate pelo **recall de `urgente`**:

| Candidato | F1 macro (val) | Recall `urgente` (val) | Tempo de treino |
|---|---|---|---|
| **Regressão Logística** (escolhido) | **0,9609** | 0,9494 | 0,6 s |
| Complement Naive Bayes | 0,9586 | 0,9551 | 0,5 s |
| Random Forest (200 árvores) | 0,9365 | 0,8989 | 2,0 s |

O Random Forest, sugerido no enunciado, ficou abaixo em qualidade e é ~70× mais lento na inferência unitária (~40 ms), o que o desqualifica para a API em tempo real.

## Uso pretendido

- **Caso de uso primário:** ordenar a fila de laudos e notas clínicas por prioridade, sinalizando os casos `urgente` para revisão imediata da equipe médica.
- **Usuários-alvo:** integração com sistemas de laudo/prontuário via API; avaliadores do Tech Challenge.
- **Escopo de entrada:** texto livre em português de exames de imagem (RX, TC, US), laboratório, ECG e notas de pronto-socorro.

### Fora de escopo

- **Decisão clínica autônoma:** o modelo prioriza a fila; não substitui a leitura do laudo nem define conduta.
- **Uso em produção sem retreino em dados reais:** foi treinado em laudos sintéticos (ver abaixo).
- **Outros idiomas ou textos fora do domínio clínico.**

## Dados de treino

- **Origem:** gerador determinístico de laudos sintéticos (`src/triage/data/generator.py`): 6 tipos de exame, ~170 achados clínicos com severidade anotada, templates de conclusão e ruído realista (abreviações, erros de digitação, variação de caixa, ausência de conclusão em 30% dos laudos, 2% de ruído de rótulo).
- **Motivo:** não existe corpus público em português com laudos livres rotulados por urgência; datasets reais com esse rótulo (MIMIC-III/IV-ED) exigem credenciamento. O gerador torna a ingestão reprodutível sem downloads e permite trocar a fonte por um CSV real com as mesmas colunas (`text`, `label`).
- **Volume e distribuição:** 6.000 laudos — `normal` 2.969 (49,5%), `atencao` 1.847 (30,8%), `urgente` 1.184 (19,7%). Comprimento médio de 232 caracteres (61 a 517).
- **Rotulagem:** regra de triagem — qualquer achado urgente ⇒ `urgente`; senão qualquer achado de atenção ⇒ `atencao`; senão `normal`.
- **Split:** treino 4.199 / validação 901 / teste 900, estratificado.
- **Validação de contrato** antes do treino (`triage.data.validation`): colunas, textos vazios, rótulos desconhecidos, presença mínima de cada classe.

## Métricas de avaliação (conjunto de teste, 900 laudos)

| Métrica | Valor |
|---|---|
| Acurácia | **0,9667** |
| F1 macro | **0,9636** |
| F1 ponderado | 0,9666 |
| Recall `urgente` | **0,9438** |

| Classe | Precisão | Recall | F1 | Suporte |
|---|---|---|---|---|
| normal | 0,967 | 0,982 | 0,974 | 445 |
| atencao | 0,967 | 0,957 | 0,962 | 277 |
| urgente | 0,966 | 0,944 | 0,955 | 178 |

Matriz de confusão (linhas = real, colunas = previsto; ordem normal / atencao / urgente):

```
            normal  atencao  urgente
normal         437        4        4
atencao         10      265        2
urgente          5        5      168
```

Dos 178 laudos urgentes, 10 foram classificados abaixo (5 como `normal`, 5 como `atencao`). Como o dataset carrega 2% de ruído de rótulo por construção (~18 amostras no teste), parte desses erros é irredutível.

### Paridade scikit-learn × ONNX (1.000 amostras)

| Concordância de rótulos | Desvio máx. de probabilidade | Desvio médio |
|---|---|---|
| **100%** | 0,0095 | 0,0004 |

### Latência

Ver [`docs/latencia.md`](latencia.md) e `reports/latency_benchmark.md`: inferência unitária in-process de 0,55 ms (scikit-learn) para 0,13 ms (ONNX) no p50.

## Quality gate de promoção

Um candidato só substitui o modelo servido se, no teste: F1 macro ≥ 0,90, recall de `urgente` ≥ 0,90, concordância ONNX ≥ 0,99 e desvio máximo de probabilidade ≤ 0,05 (`src/triage/pipelines/promote.py`).

## Limitações

1. **Dados sintéticos:** a variedade lexical é limitada pelos templates; as métricas não se transferem para laudos reais sem retreino e reavaliação.
2. **Bag-of-n-grams:** negações e contexto longo são capturados apenas parcialmente por bigramas ("sem sinais de pneumotórax" vs "pneumotórax volumoso").
3. **Sem calibração explícita:** as probabilidades da Regressão Logística são razoavelmente calibradas, mas não foram ajustadas (ex.: Platt/isotônica) para uso como limiar clínico.
4. **Classe `atencao` é a mais ambígua:** concentra a maior parte das confusões, por ser intermediária entre as outras duas.

## Considerações éticas e de segurança

- **Falsos negativos em `urgente` são o risco crítico.** A configuração prioriza recall dessa classe (`class_weight="balanced"`, gate de recall ≥ 0,90). Em produção, recomenda-se revisão humana dos casos `atencao` com baixa confiança e auditoria periódica dos `normal`.
- **Vieses:** o gerador não usa atributos sensíveis para definir o rótulo (sexo e idade aparecem apenas no cabeçalho). Em dados reais, monitorar o desempenho por tipo de exame e unidade de origem.
- **Privacidade:** a API não persiste o texto dos laudos; logs contêm apenas metadados da requisição.
- **Transparência:** a resposta inclui as probabilidades por classe, a versão do modelo e o backend, permitindo rastrear qualquer decisão até o artefato e o dataset (hashes em `metadata.json`).
