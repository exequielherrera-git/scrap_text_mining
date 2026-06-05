# -*- coding: utf-8 -*-
"""
Validación temporal con 3 cortes usando corpus enero-mayo.
  Corte 1: train=ene-mar,  test=abr-may   (split trimestral)
  Corte 2: train=ene-abr,  test=may       (split mensual)
  Corte 3: train=80%,      test=20%       (proporcional, ordenado cronológicamente)
"""
import json
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
import sys

ROOT = Path(__file__).parent
ARTIFACTS_DIR = ROOT / 'data' / 'artifacts'
RESULTS_FILE = ARTIFACTS_DIR / 'temporal_validation.json'

textos = joblib.load(ARTIFACTS_DIR / 'textos.joblib')
nombres_targets = joblib.load(ARTIFACTS_DIR / 'targets.joblib')
fechas = joblib.load(ARTIFACTS_DIR / 'fechas.joblib')

le = LabelEncoder()
targets = le.fit_transform(nombres_targets)
clases = list(le.classes_)

# Parse fechas to datetime
def parse_fecha(f):
    if not f:
        return None
    for fmt in ('%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S%z', '%Y-%m-%d'):
        try:
            return datetime.strptime(f[:19].replace('T', 'T'), fmt)
        except:
            pass
    try:
        return datetime.fromisoformat(f[:10])
    except:
        return None

fechas_dt = [parse_fecha(f) for f in fechas]
validas = [(i, dt) for i, dt in enumerate(fechas_dt) if dt is not None]
print(f"Docs totales: {len(textos)}, con fecha válida: {len(validas)}")

# Ordenar por fecha
validas_sorted = sorted(validas, key=lambda x: x[1])
indices_sorted = [x[0] for x in validas_sorted]
fechas_sorted = [x[1] for x in validas_sorted]

# Análisis por mes
from collections import Counter
meses = Counter(dt.strftime('%Y-%m') for dt in fechas_sorted)
print("Distribución por mes:", dict(sorted(meses.items())))

def corte_por_fecha(fecha_limite_str):
    """Indices de train y test según límite de fecha."""
    lim = datetime.fromisoformat(fecha_limite_str)
    train = [i for i, dt in zip(indices_sorted, fechas_sorted) if dt < lim]
    test = [i for i, dt in zip(indices_sorted, fechas_sorted) if dt >= lim]
    return train, test

def corte_proporcional(pct_train=0.8):
    """80/20 cronológico."""
    n = len(indices_sorted)
    n_train = int(n * pct_train)
    return indices_sorted[:n_train], indices_sorted[n_train:]

def entrenar_y_evaluar(train_idx, test_idx, nombre):
    print(f"\n=== {nombre} ===")
    print(f"  Train: {len(train_idx)} docs  |  Test: {len(test_idx)} docs")

    textos_train = [textos[i] for i in train_idx]
    textos_test  = [textos[i] for i in test_idx]
    y_train = targets[train_idx]
    y_test  = targets[test_idx]

    # Distribución clases en test
    from collections import Counter
    dist = Counter(clases[y] for y in y_test)
    print(f"  Test clases: {dict(dist)}")

    # TF-IDF
    vec = TfidfVectorizer(
        analyzer='word',
        lowercase=True,
        strip_accents='unicode',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(textos_train)
    X_test  = vec.transform(textos_test)

    # LSA
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train = svd.fit_transform(X_train)
    X_test  = svd.transform(X_test)

    # L2 norm
    norm = Normalizer(copy=False)
    X_train = norm.fit_transform(X_train)
    X_test  = norm.transform(X_test)

    # SVM
    clf = SVC(kernel='linear', probability=True, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    cm = confusion_matrix(y_test, preds, labels=range(len(clases)))

    # F1 por clase
    f1_clase = f1_score(y_test, preds, average=None, labels=range(len(clases)))

    print(f"  Accuracy: {acc:.4f}  |  F1 macro: {f1:.4f}")
    print(f"  F1 por clase: {dict(zip(clases, [round(x, 4) for x in f1_clase]))}")

    return {
        'n_train': len(train_idx),
        'n_test': len(test_idx),
        'accuracy': round(acc, 4),
        'f1_macro': round(f1, 4),
        'f1_por_clase': {clases[i]: round(float(f1_clase[i]), 4) for i in range(len(clases))},
        'confusion_matrix': cm.tolist(),
    }

# Corte 1: train ene-mar, test abr-may
train1, test1 = corte_por_fecha('2026-04-01')
c1 = entrenar_y_evaluar(np.array(train1), np.array(test1), 'Corte 1: train=ene-mar, test=abr-may')

# Corte 2: train ene-abr, test may
train2, test2 = corte_por_fecha('2026-05-01')
c2 = entrenar_y_evaluar(np.array(train2), np.array(test2), 'Corte 2: train=ene-abr, test=may')

# Corte 3: proporcional 80/20
train3, test3 = corte_proporcional(0.8)
c3 = entrenar_y_evaluar(np.array(train3), np.array(test3), 'Corte 3: 80% train / 20% test (cronológico)')

results = {
    'clases': clases,
    'cortes': [
        {'nombre': 'Corte 1: ene-mar / abr-may', **c1},
        {'nombre': 'Corte 2: ene-abr / may', **c2},
        {'nombre': 'Corte 3: 80/20 cronológico', **c3},
    ]
}

with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nGuardado: {RESULTS_FILE}")
