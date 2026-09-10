# ADR 003 — Otimização de latência com ONNX Runtime (quantização INT8 descartada)

**Status:** aceito · **Data:** 2026-09-10

## Contexto

A Etapa 4 exige aplicar uma técnica de otimização de latência (ONNX, quantização ou pruning) e comparar o modelo original com o otimizado. No pipeline scikit-learn, a inferência unitária é dominada pelo overhead Python do `TfidfVectorizer.transform` (regex + construção de matriz esparsa), não pela álgebra do classificador.

## Decisão

1. **Exportar o pipeline completo para ONNX** (`skl2onnx`, opset 17, `zipmap=False`) e servir com **ONNX Runtime** (`CPUExecutionProvider`, `ORT_ENABLE_ALL`, `intra_op_num_threads=1` para requisições unitárias). Tokenização, TF-IDF e classificador executam em C++ em uma única chamada.
2. **Verificar paridade** a cada exportação (concordância de rótulos e desvio de probabilidade) e bloquear a promoção se a paridade cair (`quality_gate`).
3. **Manter o backend scikit-learn** selecionável por configuração (`TRIAGE_MODEL_BACKEND=sklearn`) para comparação ao vivo e fallback.

## Alternativas avaliadas

- **Quantização dinâmica INT8** (`onnxruntime.quantization.quantize_dynamic`): o classificador é exportado como o operador `LinearClassifier` (domínio `ai.onnx.ml`), que o quantizador não cobre — o grafo saiu idêntico. Forçar a exportação com `MatMul` genérico (`black_op={"LinearClassifier"}`) tornou o modelo mais lento e a quantização falhou na inferência de tipos. Para um modelo linear com ~5 mil features, os pesos (vocabulário × 3 classes) já são desprezíveis; o ganho está no runtime, não na precisão numérica. **Descartada, com registro do experimento.**
- **Pruning de vocabulário** (`min_df`, `max_features`): já aplicado com `min_df=2`; reduções maiores degradam o recall de `urgente` sem ganho mensurável de latência no ONNX.

## Consequências

- (+) Speedup de ~4–5× na inferência unitária in-process e redução proporcional da latência HTTP, mantendo 100% de concordância de rótulos.
- (+) Artefato portátil (o mesmo `.onnx` roda em qualquer runtime ONNX).
- (−) Restrições de exportação: sem transformadores customizados no pipeline, `token_pattern` explícito, normalização fora do pipeline.
