# ADR 002 — Modelo leve: TF-IDF + classificador linear (em vez de transformer)

**Status:** aceito · **Data:** 2026-09-10

## Contexto

O enunciado pede um classificador de texto **leve** servido em CPU, com foco no ciclo de vida (CI/CD, orquestração, monitoramento, latência). Um modelo baseado em transformer (ex.: BERTimbau) daria maior capacidade semântica, porém com latência de dezenas a centenas de milissegundos em CPU, imagens de vários GB e retreino caro.

## Decisão

Usar **TF-IDF (1-2 gramas, `sublinear_tf`) + Regressão Logística** com `class_weight="balanced"`, selecionado entre três candidatos (Regressão Logística, Complement Naive Bayes e Random Forest) por **F1 macro na validação**, com desempate pelo **recall da classe `urgente`**.

A normalização de texto (minúsculas, remoção de acentos, separador decimal) fica **fora** do pipeline do scikit-learn, em `triage.nlp.normalize_text`, aplicada identicamente no treino e na API. O `token_pattern` é explícito (`[a-z0-9][a-z0-9]+`) para que a tokenização no ONNX Runtime seja idêntica à do scikit-learn.

## Consequências

- (+) Inferência unitária < 1 ms em CPU; artefato de ~230 KB; treino em segundos (retreino diário barato).
- (+) Totalmente conversível para ONNX (`skl2onnx`), inclusive o vetorizador.
- (+) Interpretável: pesos por n-grama explicam a decisão.
- (−) Sem compreensão de contexto longo/negação sofisticada; n-gramas de 2 mitigam parcialmente ("sem sinais de pneumotórax").
- (−) Random Forest, sugerido no enunciado, ficou como candidato: mais lento (~40 ms por laudo) e com F1 inferior no dataset.
