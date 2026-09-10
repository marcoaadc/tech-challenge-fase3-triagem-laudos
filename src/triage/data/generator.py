"""Gerador deterministico de laudos medicos sinteticos em portugues, rotulados por urgencia.

Por que sintetico?
    Nao existe dataset publico em portugues com laudos livres rotulados por prioridade de
    triagem; os corpora reais com esse tipo de rotulo (ex.: MIMIC-III/IV-ED) exigem
    credenciamento e acordo de uso de dados. O gerador abaixo produz laudos plausiveis a
    partir de templates clinicos (radiografia, tomografia, laboratorio, ECG, ultrassom e
    nota de pronto-socorro), combinando achados de diferentes severidades com ruido
    realista (abreviacoes, erros de digitacao, negacoes e ausencia de conclusao).

O rotulo segue a regra de triagem: qualquer achado *urgente* => ``urgente``; senao qualquer
achado de *atencao* => ``atencao``; senao ``normal``. Um pequeno ruido de rotulo (2%) simula a
discordancia entre avaliadores e evita que o problema seja trivialmente separavel.

O gerador e uma funcao pura de ``seed``: o mesmo seed produz exatamente o mesmo CSV,
o que torna a etapa de ingestao do pipeline reprodutivel sem depender de downloads.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

LABELS: tuple[str, ...] = ("normal", "atencao", "urgente")

# Distribuicao alvo das classes (desbalanceada, como numa triagem real).
CLASS_WEIGHTS: dict[str, float] = {"normal": 0.50, "atencao": 0.30, "urgente": 0.20}

# ---------------------------------------------------------------------------
# Vocabulario clinico por tipo de exame e severidade
# ---------------------------------------------------------------------------
EXAMS: dict[str, dict[str, list[str]]] = {
    "radiografia_torax": {
        "header": [
            "RADIOGRAFIA DE TÓRAX EM PA E PERFIL.",
            "RX DE TÓRAX (PA/PERFIL). Técnica adequada.",
            "Radiografia simples de tórax, incidências PA e perfil, em inspiração.",
            "Rx tórax AP no leito.",
        ],
        "normal": [
            "Campos pulmonares com transparência preservada.",
            "Seios costofrênicos livres.",
            "Área cardíaca dentro dos limites da normalidade.",
            "Mediastino centrado, sem alargamentos.",
            "Sem sinais de pneumotórax ou derrame pleural.",
            "Estruturas ósseas íntegras.",
            "Cúpulas diafragmáticas de contornos regulares.",
            "Hilos pulmonares de aspecto habitual.",
            "Trama vascular de distribuição normal.",
            "Ausência de consolidações ou nódulos.",
            "Traqueia centrada e pérvia.",
        ],
        "atencao": [
            "Discreto espessamento peribrônquico bilateral.",
            "Opacidade reticular tênue em base direita, a correlacionar com quadro clínico.",
            "Pequeno derrame pleural à esquerda.",
            "Aumento discreto da área cardíaca (índice cardiotorácico limítrofe).",
            "Atelectasia laminar em base esquerda.",
            "Nódulo pulmonar de 7 mm em lobo superior direito, sugere-se controle tomográfico.",
            "Sinais de hiperinsuflação pulmonar compatíveis com DPOC.",
            "Elevação da cúpula diafragmática direita.",
            "Fratura consolidada de arco costal lateral esquerdo.",
            "Espessamento pleural apical bilateral, provavelmente sequelar.",
            "Calcificações ganglionares hilares, de aspecto residual.",
        ],
        "urgente": [
            "Pneumotórax volumoso à direita com desvio contralateral do mediastino.",
            "Extensa consolidação em lobo inferior esquerdo com broncogramas aéreos, compatível com pneumonia.",
            "Derrame pleural volumoso à esquerda com colapso pulmonar subjacente.",
            "Alargamento mediastinal com suspeita de dissecção de aorta.",
            "Infiltrado alveolar difuso bilateral, sugerindo edema agudo de pulmão.",
            "Pneumoperitônio: ar livre sob as cúpulas diafragmáticas.",
            "Fraturas múltiplas de arcos costais com tórax instável.",
            "Opacidades bilaterais extensas em vidro fosco, padrão de SDRA.",
            "Massa pulmonar de 6 cm em lobo superior esquerdo com invasão de parede torácica.",
            "Pneumotórax hipertensivo à esquerda.",
        ],
    },
    "tomografia_cranio": {
        "header": [
            "TOMOGRAFIA COMPUTADORIZADA DE CRÂNIO SEM CONTRASTE.",
            "TC de crânio, cortes axiais, sem administração de contraste endovenoso.",
            "TC CRÂNIO S/ CONTRASTE. Técnica: aquisição helicoidal.",
        ],
        "normal": [
            "Parênquima encefálico com coeficientes de atenuação preservados.",
            "Sistema ventricular de dimensões e morfologia normais.",
            "Ausência de coleções extra-axiais.",
            "Linha média centrada.",
            "Sem sinais de sangramento intracraniano.",
            "Cisternas da base pérvias.",
            "Calota craniana íntegra.",
            "Sulcos corticais de amplitude normal para a faixa etária.",
            "Fossa posterior sem alterações.",
        ],
        "atencao": [
            "Hipoatenuação da substância branca periventricular, compatível com microangiopatia.",
            "Discreta acentuação dos sulcos corticais, sugerindo redução volumétrica.",
            "Pequena calcificação em plexo coroide, sem significado patológico.",
            "Sinusopatia maxilar bilateral.",
            "Cisto de retenção em seio maxilar esquerdo.",
            "Lacuna isquêmica antiga em núcleo lentiforme direito.",
            "Pequeno higroma subdural frontal, sem efeito de massa.",
            "Alargamento discreto do sistema ventricular, a correlacionar clinicamente.",
        ],
        "urgente": [
            "Hematoma subdural agudo fronto-parietal direito com desvio de linha média de 9 mm.",
            "Hemorragia intraparenquimatosa em núcleos da base à esquerda com inundação ventricular.",
            "Extensa área de hipoatenuação em território de artéria cerebral média direita, compatível com AVC isquêmico agudo.",
            "Hemorragia subaracnoidea difusa nas cisternas da base.",
            "Hematoma epidural temporal esquerdo com efeito de massa.",
            "Hidrocefalia aguda com transudação ependimária.",
            "Lesão expansiva frontal com edema perilesional e herniação subfalcina.",
            "Fratura com afundamento de osso parietal e contusão hemorrágica subjacente.",
            "Sinais de edema cerebral difuso com apagamento das cisternas da base.",
        ],
    },
    "laboratorio": {
        "header": [
            "EXAMES LABORATORIAIS. Material: sangue venoso.",
            "HEMOGRAMA E BIOQUÍMICA. Coleta em jejum.",
            "Resultados de exames laboratoriais (soro/plasma).",
            "Painel laboratorial de urgência.",
        ],
        "normal": [
            "Hemoglobina: 13,8 g/dL (VR 12,0-16,0).",
            "Leucócitos: 6.900/mm3, sem desvio.",
            "Plaquetas: 245.000/mm3.",
            "Creatinina: 0,9 mg/dL. Ureia: 32 mg/dL.",
            "Potássio: 4,1 mmol/L. Sódio: 139 mmol/L.",
            "Glicemia de jejum: 91 mg/dL.",
            "PCR: 0,4 mg/dL.",
            "TGO/TGP dentro dos valores de referência.",
            "TSH: 2,1 mUI/L.",
            "Troponina I: indetectável.",
            "Lactato: 1,1 mmol/L.",
        ],
        "atencao": [
            "Hemoglobina: 10,9 g/dL, anemia leve normocítica.",
            "Leucócitos: 12.800/mm3 com discreto desvio à esquerda.",
            "Glicemia de jejum: 131 mg/dL, sugerindo hiperglicemia; recomenda-se HbA1c.",
            "Creatinina: 1,6 mg/dL, discreta elevação em relação ao basal.",
            "Potássio: 5,4 mmol/L, discretamente elevado.",
            "PCR: 4,8 mg/dL, elevada.",
            "TGP: 96 U/L, elevação de transaminases.",
            "Colesterol LDL: 178 mg/dL.",
            "Plaquetas: 118.000/mm3, plaquetopenia leve.",
            "TSH: 8,4 mUI/L, hipotireoidismo subclínico.",
            "Vitamina D: 14 ng/mL, deficiência.",
        ],
        "urgente": [
            "Potássio: 6,9 mmol/L. VALOR CRÍTICO.",
            "Troponina I: 2,40 ng/mL, acentuadamente elevada.",
            "Glicemia: 512 mg/dL com cetonemia positiva.",
            "Hemoglobina: 5,8 g/dL, anemia grave.",
            "Leucócitos: 31.000/mm3 com 18% de bastonetes.",
            "Lactato: 6,2 mmol/L. Gasometria: pH 7,18, HCO3 12.",
            "Plaquetas: 14.000/mm3. RISCO DE SANGRAMENTO.",
            "Creatinina: 6,8 mg/dL, ureia 214 mg/dL, insuficiência renal aguda.",
            "Sódio: 118 mmol/L, hiponatremia grave.",
            "Hemocultura positiva para Staphylococcus aureus em 2 amostras.",
            "INR: 8,2 em paciente anticoagulado.",
        ],
    },
    "ecg": {
        "header": [
            "ELETROCARDIOGRAMA DE 12 DERIVAÇÕES.",
            "ECG de repouso, 12 derivações, 25 mm/s.",
            "ECG realizado no leito.",
        ],
        "normal": [
            "Ritmo sinusal, frequência cardíaca de 72 bpm.",
            "Eixo elétrico normal.",
            "Intervalo PR de 160 ms.",
            "QRS estreito, 88 ms.",
            "QTc de 410 ms.",
            "Sem alterações da repolarização ventricular.",
            "Ausência de arritmias.",
            "Onda P de morfologia normal.",
        ],
        "atencao": [
            "Bradicardia sinusal, FC 52 bpm.",
            "Bloqueio de ramo direito incompleto.",
            "Extrassístoles ventriculares isoladas.",
            "Sobrecarga ventricular esquerda pelos critérios de Sokolow-Lyon.",
            "Alterações inespecíficas da repolarização ventricular em parede lateral.",
            "Bloqueio atrioventricular de primeiro grau, PR 240 ms.",
            "Fibrilação atrial com resposta ventricular controlada, FC 84 bpm.",
            "QTc de 470 ms, limítrofe.",
        ],
        "urgente": [
            "Supradesnivelamento do segmento ST de 3 mm em V1-V4, compatível com IAM anterior.",
            "Taquicardia ventricular sustentada, FC 180 bpm.",
            "Bloqueio atrioventricular total com ritmo de escape a 34 bpm.",
            "Fibrilação atrial com alta resposta ventricular, FC 168 bpm.",
            "Infradesnivelamento difuso de ST com supra em aVR, sugerindo lesão de tronco.",
            "QTc de 560 ms com episódios de torsades de pointes.",
            "Ondas T apiculadas difusas e alargamento do QRS, sugestivo de hipercalemia grave.",
            "Padrão de Brugada tipo 1 em paciente com síncope.",
        ],
    },
    "ultrassom_abdome": {
        "header": [
            "ULTRASSONOGRAFIA DE ABDOME TOTAL.",
            "US abdome total, transdutor convexo.",
            "Ultrassom de abdome superior e pelve.",
        ],
        "normal": [
            "Fígado de dimensões normais e ecotextura homogênea.",
            "Vesícula biliar de paredes finas, sem cálculos.",
            "Vias biliares intra e extra-hepáticas não dilatadas.",
            "Rins tópicos, de dimensões e espessura parenquimatosa preservadas.",
            "Baço de dimensões normais.",
            "Ausência de líquido livre na cavidade.",
            "Aorta abdominal de calibre normal.",
            "Bexiga com paredes regulares.",
            "Pâncreas de aspecto habitual, no segmento visualizado.",
        ],
        "atencao": [
            "Esteatose hepática grau II.",
            "Colelitíase: cálculos móveis de até 8 mm, sem sinais de colecistite.",
            "Cisto renal simples de 2,3 cm em polo superior direito (Bosniak I).",
            "Discreta esplenomegalia (13,5 cm).",
            "Microlitíase renal bilateral, sem dilatação do sistema coletor.",
            "Hemangioma hepático típico de 1,8 cm em segmento VI.",
            "Próstata aumentada de volume (52 cm3).",
            "Pequena quantidade de líquido livre em fundo de saco, a correlacionar.",
        ],
        "urgente": [
            "Vesícula biliar distendida com paredes espessadas (6 mm), cálculo impactado no infundíbulo e Murphy ultrassonográfico positivo: colecistite aguda.",
            "Grande quantidade de líquido livre na cavidade com ecos em suspensão, sugerindo hemoperitônio.",
            "Aneurisma de aorta abdominal de 6,2 cm com trombo mural.",
            "Hidronefrose acentuada à esquerda por cálculo obstrutivo ureteral de 12 mm.",
            "Apêndice cecal espessado (11 mm), não compressível, com líquido periapendicular: apendicite aguda.",
            "Coleção hepática de 9 cm com debris, compatível com abscesso.",
            "Gestação ectópica tubária rota com hemoperitônio.",
            "Dilatação acentuada de vias biliares com cálculo em colédoco distal e sinais de colangite.",
        ],
    },
    "nota_pronto_socorro": {
        "header": [
            "NOTA DE TRIAGEM - PRONTO-SOCORRO.",
            "Ficha de acolhimento com classificação de risco.",
            "Evolução de enfermagem na admissão.",
            "Anamnese de admissão no PS.",
        ],
        "normal": [
            "Paciente refere dor de garganta há 2 dias, sem febre.",
            "Sinais vitais: PA 120x80 mmHg, FC 76 bpm, FR 16 irpm, SatO2 98% em ar ambiente, Tax 36,6 C.",
            "Queixa de coriza e tosse seca, sem dispneia.",
            "Solicita renovação de receita de uso contínuo.",
            "Dor lombar mecânica leve após esforço, sem irradiação ou déficit.",
            "Glasgow 15, orientado, deambulando.",
            "Retorno para reavaliação de curativo, ferida limpa e sem sinais flogísticos.",
            "Nega comorbidades. Exame físico sem alterações.",
        ],
        "atencao": [
            "Febre de 38,5 C há 3 dias com tosse produtiva.",
            "PA 158x98 mmHg, assintomático, com histórico de hipertensão.",
            "Dor abdominal em cólica há 12 horas, sem sinais de irritação peritoneal.",
            "Vômitos há 24h com tolerância parcial à dieta, sem sinais de desidratação grave.",
            "Glicemia capilar de 289 mg/dL em paciente diabético, sem cetose.",
            "Crise de asma leve, SatO2 94%, respondendo a broncodilatador.",
            "Ferimento cortocontuso em mão direita, com necessidade de sutura.",
            "Cefaleia tensional recorrente, sem sinais de alarme.",
            "Dor torácica atípica, ECG sem alterações agudas, troponina pendente.",
        ],
        "urgente": [
            "Dor torácica opressiva há 40 minutos com irradiação para MSE e sudorese fria.",
            "SatO2 84% em ar ambiente, FR 32 irpm, uso de musculatura acessória.",
            "Rebaixamento do nível de consciência, Glasgow 8.",
            "PA 70x40 mmHg, FC 132 bpm, extremidades frias, tempo de enchimento capilar > 3 s.",
            "Déficit motor súbito em dimídio direito com afasia, início há 1 hora.",
            "Crise convulsiva tônico-clônica em atividade há mais de 5 minutos.",
            "Hemorragia digestiva alta com hematêmese volumosa e instabilidade hemodinâmica.",
            "Anafilaxia: edema de glote, estridor e hipotensão após uso de dipirona.",
            "Trauma cranioencefálico grave com anisocoria.",
            "Febre de 40 C, hipotensão e rigidez de nuca.",
        ],
    },
}

IMPRESSIONS: dict[str, list[str]] = {
    "normal": [
        "Impressão diagnóstica: exame sem alterações significativas.",
        "Conclusão: estudo dentro dos padrões da normalidade.",
        "Sem achados agudos.",
        "Exame normal.",
        "Conclusão: sem evidência de doença aguda. Seguimento de rotina.",
    ],
    "atencao": [
        "Impressão: achados que merecem acompanhamento ambulatorial.",
        "Sugere-se correlação clínico-laboratorial e reavaliação em 30 dias.",
        "Conclusão: alterações sem caráter de urgência; recomenda-se seguimento com especialista.",
        "Achados de baixa gravidade, orientar retorno eletivo.",
        "Recomenda-se controle evolutivo.",
    ],
    "urgente": [
        "ACHADO CRÍTICO - comunicado ao médico solicitante.",
        "Conclusão: alterações agudas que requerem avaliação imediata.",
        "Recomenda-se encaminhamento imediato à emergência.",
        "Prioridade máxima. Equipe médica acionada.",
        "Achados com risco iminente de vida.",
    ],
}

PATIENT_TEMPLATES = [
    "Paciente {sexo}, {idade} anos.",
    "Pcte {sexo_abrev}, {idade}a.",
    "Identificação: {sexo}, {idade} anos.",
    "",
]

ABBREVIATIONS = {
    "paciente": "pcte",
    "sem ": "s/ ",
    "com ": "c/ ",
    "direita": "D",
    "esquerda": "E",
    "bilateral": "bilat.",
    "frequência cardíaca": "FC",
    "compatível com": "c/c",
    "recomenda-se": "rec.",
}


@dataclass(frozen=True)
class GeneratedReport:
    laudo_id: str
    exam_type: str
    text: str
    label: str


def _severity_plan(rng: random.Random, label: str) -> tuple[int, int, int]:
    """Quantos achados (normal, atencao, urgente) compor para a classe alvo."""
    if label == "normal":
        return rng.randint(2, 4), 0, 0
    if label == "atencao":
        return rng.randint(0, 3), rng.randint(1, 2), 0
    return rng.randint(0, 2), rng.randint(0, 1), rng.randint(1, 2)


def _typo(rng: random.Random, word: str) -> str:
    """Troca dois caracteres adjacentes (erro de digitacao comum)."""
    if len(word) < 5:
        return word
    i = rng.randrange(len(word) - 1)
    return word[:i] + word[i + 1] + word[i] + word[i + 2 :]


def _add_noise(rng: random.Random, text: str) -> str:
    # abreviacoes clinicas
    if rng.random() < 0.35:
        for full, abbr in ABBREVIATIONS.items():
            if rng.random() < 0.5:
                text = text.replace(full, abbr)
    # erros de digitacao esparsos
    if rng.random() < 0.4:
        words = text.split(" ")
        n_typos = max(1, len(words) // 25)
        for _ in range(n_typos):
            j = rng.randrange(len(words))
            words[j] = _typo(rng, words[j])
        text = " ".join(words)
    # variacao de caixa
    roll = rng.random()
    if roll < 0.10:
        text = text.upper()
    elif roll < 0.20:
        text = text.lower()
    # remocao parcial de pontuacao
    if rng.random() < 0.15:
        text = text.replace(".", "")
    return text


def _compose(rng: random.Random, exam_type: str, label: str) -> str:
    exam = EXAMS[exam_type]
    n_normal, n_atencao, n_urgente = _severity_plan(rng, label)
    findings = (
        rng.sample(exam["normal"], min(n_normal, len(exam["normal"])))
        + rng.sample(exam["atencao"], min(n_atencao, len(exam["atencao"])))
        + rng.sample(exam["urgente"], min(n_urgente, len(exam["urgente"])))
    )
    rng.shuffle(findings)

    sexo = rng.choice(["masculino", "feminino"])
    patient = rng.choice(PATIENT_TEMPLATES).format(
        sexo=sexo, sexo_abrev="M" if sexo == "masculino" else "F", idade=rng.randint(1, 95)
    )
    parts = [rng.choice(exam["header"])]
    if patient:
        parts.append(patient)
    parts.extend(findings)
    # 30% dos laudos nao trazem conclusao: o modelo precisa aprender pelos achados.
    if rng.random() < 0.70:
        parts.append(rng.choice(IMPRESSIONS[label]))

    separator = rng.choice([" ", " ", "\n", " - "])
    return _add_noise(rng, separator.join(parts))


def generate_dataset(n_samples: int = 6000, seed: int = 42, label_noise: float = 0.02) -> list[GeneratedReport]:
    """Gera ``n_samples`` laudos rotulados, de forma deterministica para um ``seed``."""
    if n_samples <= 0:
        raise ValueError("n_samples deve ser positivo")
    if not 0.0 <= label_noise < 1.0:
        raise ValueError("label_noise deve estar em [0, 1)")

    rng = random.Random(seed)
    exam_types = list(EXAMS)
    labels, weights = zip(*CLASS_WEIGHTS.items(), strict=True)
    reports: list[GeneratedReport] = []
    for i in range(n_samples):
        exam_type = rng.choice(exam_types)
        label = rng.choices(labels, weights=weights, k=1)[0]
        text = _compose(rng, exam_type, label)
        if rng.random() < label_noise:
            label = rng.choice([lb for lb in LABELS if lb != label])
        reports.append(GeneratedReport(laudo_id=f"L{i + 1:06d}", exam_type=exam_type, text=text, label=label))
    return reports


def write_csv(reports: list[GeneratedReport], path: Path) -> Path:
    """Persiste os laudos em CSV (UTF-8) com as colunas esperadas pelo pipeline de treino."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["laudo_id", "exam_type", "text", "label"])
        for r in reports:
            writer.writerow([r.laudo_id, r.exam_type, r.text, r.label])
    return path
