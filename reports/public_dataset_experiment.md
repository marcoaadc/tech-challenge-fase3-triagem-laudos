# Experimento em dataset publico - Medical Abstracts TC Corpus

Fonte: <https://github.com/sebischair/Medical-Abstracts-TC-Corpus> (en), treino 11550 / teste 2888, 5 classes: cardiovascular_diseases, digestive_system_diseases, general_pathological_conditions, neoplasms, nervous_system_diseases.

Mesmo pipeline do projeto (TF-IDF 1-2 gramas + candidatos), sem ajuste de hiperparametros.
Baseline da classe majoritaria (`general_pathological_conditions`): acuracia 0.3328.

| Candidato | Acuracia | F1 macro | F1 ponderado | Vocabulario | Treino (s) |
|---|---|---|---|---|---|
| logreg | 0.5118 | 0.5104 | 0.5089 | 233677 | 31.7 |
| complement_nb (melhor) | 0.5419 | 0.5182 | 0.5249 | 233677 | 6.5 |

Paridade scikit-learn x ONNX (1000 amostras): concordancia de rotulos 0.9980, desvio maximo de probabilidade 0.0110.

Calibracao do melhor modelo (teste): Brier 0.6332, ECE 0.1204, MCE 0.2359.

Citacao: Schopf, T., Braun, D., Matthes, F. (2022). Evaluating Unsupervised Text Classification: Zero-shot and Similarity-based Approaches. NLPIR 2022.
