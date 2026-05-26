# -*- coding: utf-8 -*-
"""
TF-IDF + LSA (Latent Semantic Analysis) + SVM (kernel lineal).

Alternativa densa al BoW: en lugar de contar palabras (sparse, ~62k dims),
TF-IDF pondera la relevancia de cada término y TruncatedSVD reduce el espacio
a 100 dimensiones capturando co-ocurrencias semánticas — enfoque equivalente
al de Word2Vec en términos de producir vectores densos por documento.

Ventaja frente a Word2Vec puro: no requiere dependencias externas (gensim),
corre sobre sklearn y numpy disponibles en el entorno estándar del proyecto.

Flujo:
  textos.joblib
  →  TfidfVectorizer  (sparse, ~62k dims)
  →  TruncatedSVD     (dense, 100 dims)   ← LSA
  →  SVM kernel lineal, 5-fold CV estratificado

Artefactos de entrada  (data/artifacts/):
    textos.joblib, targets.joblib

Artefactos de salida  (data/artifacts/):
    modelo_w2v.joblib    - pipeline TF-IDF + SVD + SVM final + label_encoder
    metricas_w2v.json    - métricas de cross-validation para el dashboard
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, label_binarize
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).parent))
from tokenizadores import tokenizador

_ROOT         = Path(__file__).parent.parent
ARTIFACTS_DIR = _ROOT / 'data' / 'artifacts'

N_COMPONENTS: int = 100   # dimensiones del espacio LSA
N_FOLDS:      int = 5

METRICAS_FILE = ARTIFACTS_DIR / 'metricas_w2v.json'
MODELO_FILE   = ARTIFACTS_DIR / 'modelo_w2v.joblib'

# Ruta a stopwords (misma lógica que features.py)
_SW_SIN_ACENTOS = _ROOT / 'resources' / 'stopwords_es_sin_acentos.txt'
_SW_CON_ACENTOS = _ROOT / 'resources' / 'stopwords_es.txt'


def _leer_stopwords() -> list[str]:
    for p in (_SW_SIN_ACENTOS, _SW_CON_ACENTOS):
        if p.exists():
            return [w.strip().lower() for w in p.read_text(encoding='utf-8').splitlines() if w.strip()]
    return []


def _auc_por_clase(y_bin: np.ndarray, proba: np.ndarray, n_clases: int) -> list[float]:
    result = []
    for i in range(n_clases):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            result.append(float(auc(fpr, tpr)))
        except Exception:
            result.append(0.0)
    return result


if __name__ == '__main__':
    needed = ['textos.joblib', 'targets.joblib']
    missing = [f for f in needed if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        print(f'ERROR: faltan artefactos: {", ".join(missing)}')
        print('Ejecutá primero la vectorización (paso 2 del pipeline).')
        sys.exit(1)

    print('Cargando corpus...')
    textos          = joblib.load(ARTIFACTS_DIR / 'textos.joblib')
    nombres_targets = joblib.load(ARTIFACTS_DIR / 'targets.joblib')

    label_encoder = LabelEncoder()
    targets  = label_encoder.fit_transform(nombres_targets)
    clases   = list(label_encoder.classes_)
    n_clases = len(clases)

    stopwords  = _leer_stopwords()
    tokenize   = tokenizador()

    print(f'Corpus: {len(textos)} docs  |  {n_clases} clases: {", ".join(clases)}')
    print(f'Representacion: TF-IDF + TruncatedSVD ({N_COMPONENTS} dims) - LSA')
    print(f'Clasificador: SVM kernel lineal  |  CV: {N_FOLDS} folds')
    print()

    # Pipeline de vectorización: TF-IDF → SVD → normalización L2
    # (normalizar después de SVD mejora la separabilidad con kernel lineal)
    tfidf = TfidfVectorizer(
        tokenizer=tokenize,
        stop_words=stopwords,
        lowercase=True,
        strip_accents='unicode',
        decode_error='ignore',
        ngram_range=(1, 1),   # unigramas: más compatibles con SVD semántico
        min_df=2,
        max_df=0.9,
    )
    svd  = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
    norm = Normalizer(copy=False)

    print('Computando TF-IDF...')
    t0 = time.time()
    X_tfidf = tfidf.fit_transform(textos)
    elapsed_tfidf = time.time() - t0
    print(f'TF-IDF listo: {X_tfidf.shape[0]} docs × {X_tfidf.shape[1]} features  ({elapsed_tfidf:.1f}s)')

    print(f'Aplicando TruncatedSVD ({N_COMPONENTS} componentes)...')
    t1 = time.time()
    X = norm.fit_transform(svd.fit_transform(X_tfidf))
    elapsed_svd = time.time() - t1
    var_explained = svd.explained_variance_ratio_.sum()
    print(f'LSA listo: {X.shape[0]} docs × {X.shape[1]} dims  |  varianza explicada: {var_explained:.1%}  ({elapsed_svd:.1f}s)')
    print()

    targets_bin  = label_binarize(targets, classes=range(n_clases))
    clasificador = SVC(kernel='linear', probability=True, random_state=42)
    kf           = StratifiedKFold(n_splits=N_FOLDS, random_state=42, shuffle=True)

    accuracy_folds:     list[float]       = []
    auc_folds:          list[list[float]] = []
    conf_matrix_total = np.zeros((n_clases, n_clases), dtype=int)

    t_cv = time.time()

    for fold_n, (train_idx, test_idx) in enumerate(kf.split(X, targets), 1):
        print(f'-- Fold {fold_n}/{N_FOLDS} ----------------------------------')
        print(f'  Train: {len(train_idx)} docs  |  Test: {len(test_idx)} docs')

        clasificador.fit(X[train_idx], targets[train_idx])
        preds = clasificador.predict(X[test_idx])
        proba = clasificador.predict_proba(X[test_idx])

        acc = accuracy_score(targets[test_idx], preds)
        accuracy_folds.append(acc)
        print(f'  Accuracy: {acc:.4f}')

        aucs = _auc_por_clase(targets_bin[test_idx], proba, n_clases)
        auc_folds.append(aucs)
        for i, clase in enumerate(clases):
            print(f'  AUC {clase}: {aucs[i]:.4f}')

        conf_matrix_total += confusion_matrix(targets[test_idx], preds, labels=range(n_clases))
        print()

    elapsed_cv = time.time() - t_cv
    elapsed    = (time.time() - t0)
    acc_mean   = float(np.mean(accuracy_folds))
    acc_std    = float(np.std(accuracy_folds))
    auc_mean   = [float(np.mean([f[i] for f in auc_folds])) for i in range(n_clases)]

    print('== Resultados finales ==================================')
    print(f'Accuracy promedio: {acc_mean:.4f} ± {acc_std:.4f}')
    for i, clase in enumerate(clases):
        print(f'AUC {clase}: {auc_mean[i]:.4f}')
    print(f'Varianza LSA explicada: {var_explained:.1%}')
    print(f'Tiempo TF-IDF: {elapsed_tfidf:.1f}s  |  SVD: {elapsed_svd:.1f}s  |  CV: {elapsed_cv:.1f}s  |  Total: {elapsed:.1f}s')
    print()

    # Modelo final sobre dataset completo
    print('Entrenando modelo final sobre dataset completo...')
    X_final      = norm.fit_transform(svd.fit_transform(tfidf.transform(textos)))
    modelo_final = SVC(kernel='linear', probability=True, random_state=42)
    modelo_final.fit(X_final, targets)

    metricas = {
        'clasificador':    'TF-IDF + LSA + SVM (kernel lineal)',
        'features_method': f'TF-IDF → SVD ({N_COMPONENTS} dims) — LSA',
        'n_folds':         N_FOLDS,
        'n_docs':          len(textos),
        'n_features_orig': X_tfidf.shape[1],
        'n_features_sel':  N_COMPONENTS,
        'clases':          clases,
        'accuracy_folds':  [round(a, 4) for a in accuracy_folds],
        'accuracy_mean':   round(acc_mean, 4),
        'accuracy_std':    round(acc_std, 4),
        'auc_por_clase':   {clases[i]: round(auc_mean[i], 4) for i in range(n_clases)},
        'confusion_matrix': conf_matrix_total.tolist(),
        'elapsed_s':       round(elapsed, 1),
        'lsa_components':  N_COMPONENTS,
        'lsa_var_explained': round(float(var_explained), 4),
    }

    with open(METRICAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {'modelo': modelo_final, 'tfidf': tfidf, 'svd': svd,
         'normalizer': norm, 'label_encoder': label_encoder},
        MODELO_FILE,
    )

    print(f'Modelo guardado:    {MODELO_FILE}')
    print(f'Métricas guardadas: {METRICAS_FILE}')
