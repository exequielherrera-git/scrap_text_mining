# -*- coding: utf-8 -*-
"""
Entrena un clasificador SVM (kernel lineal) con selección de features chi2.

Lee los artefactos generados por pipeline/features.py, realiza cross-validation
estratificado (5 folds) y guarda el modelo final + métricas en data/artifacts/.

Artefactos de entrada  (data/artifacts/):
    vectores.joblib, targets.joblib, features.joblib

Artefactos de salida  (data/artifacts/):
    modelo_svm.joblib   - modelo final entrenado sobre el dataset completo
    metricas_svm.json   - métricas de cross-validation para el dashboard
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC

_ROOT         = Path(__file__).parent.parent
ARTIFACTS_DIR = _ROOT / 'data' / 'artifacts'

MAX_FEATURES:  int = 150   # features seleccionadas por chi2
N_FOLDS:       int = 5     # folds para cross-validation estratificado
METRICAS_FILE      = ARTIFACTS_DIR / 'metricas_svm.json'
MODELO_FILE        = ARTIFACTS_DIR / 'modelo_svm.joblib'


def _auc_por_clase(y_bin: np.ndarray, proba: np.ndarray, n_clases: int) -> list[float]:
    """AUC OvR para cada clase usando predict_proba del SVC."""
    result = []
    for i in range(n_clases):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            result.append(float(auc(fpr, tpr)))
        except Exception:
            result.append(0.0)
    return result


if __name__ == '__main__':
    needed = ['vectores.joblib', 'targets.joblib', 'features.joblib']
    missing = [f for f in needed if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        print(f'ERROR: faltan artefactos: {", ".join(missing)}')
        print('Ejecutá primero la vectorización (paso 2 del pipeline).')
        sys.exit(1)

    print('Cargando artefactos...')
    vectores        = joblib.load(ARTIFACTS_DIR / 'vectores.joblib')
    nombres_targets = joblib.load(ARTIFACTS_DIR / 'targets.joblib')

    label_encoder = LabelEncoder()
    targets  = label_encoder.fit_transform(nombres_targets)
    clases   = list(label_encoder.classes_)
    n_clases = len(clases)

    n_docs, n_feat = vectores.shape
    print(f'Dataset: {n_docs} docs × {n_feat} features  |  {n_clases} clases: {", ".join(clases)}')
    print(f'Selección: chi2 (k={MAX_FEATURES})  |  Clasificador: SVM kernel lineal  |  CV: {N_FOLDS} folds')
    print()

    targets_bin  = label_binarize(targets, classes=range(n_clases))
    clasificador = SVC(kernel='linear', probability=True, random_state=42)
    kf           = StratifiedKFold(n_splits=N_FOLDS, random_state=42, shuffle=True)

    accuracy_folds:     list[float]       = []
    auc_folds:          list[list[float]] = []
    conf_matrix_total = np.zeros((n_clases, n_clases), dtype=int)

    t0 = time.time()

    for fold_n, (train_idx, test_idx) in enumerate(kf.split(vectores, targets), 1):
        print(f'── Fold {fold_n}/{N_FOLDS} ──────────────────────────────────')
        print(f'  Train: {len(train_idx)} docs  |  Test: {len(test_idx)} docs')

        selector    = SelectKBest(score_func=chi2, k=MAX_FEATURES)
        train_X_sel = selector.fit_transform(vectores[train_idx], targets[train_idx])
        test_X_sel  = selector.transform(vectores[test_idx])

        clasificador.fit(train_X_sel, targets[train_idx])
        preds = clasificador.predict(test_X_sel)
        proba = clasificador.predict_proba(test_X_sel)

        acc = accuracy_score(targets[test_idx], preds)
        accuracy_folds.append(acc)
        print(f'  Accuracy: {acc:.4f}')

        aucs = _auc_por_clase(targets_bin[test_idx], proba, n_clases)
        auc_folds.append(aucs)
        for i, clase in enumerate(clases):
            print(f'  AUC {clase}: {aucs[i]:.4f}')

        conf_matrix_total += confusion_matrix(targets[test_idx], preds, labels=range(n_clases))
        print()

    elapsed  = time.time() - t0
    acc_mean = float(np.mean(accuracy_folds))
    acc_std  = float(np.std(accuracy_folds))
    auc_mean = [float(np.mean([f[i] for f in auc_folds])) for i in range(n_clases)]

    print('══ Resultados finales ══════════════════════════════')
    print(f'Accuracy promedio: {acc_mean:.4f} ± {acc_std:.4f}')
    for i, clase in enumerate(clases):
        print(f'AUC {clase}: {auc_mean[i]:.4f}')
    print(f'Tiempo total de CV: {elapsed:.1f}s')
    print()

    # Modelo final entrenado sobre el dataset completo (para predicciones futuras)
    print('Entrenando modelo final sobre dataset completo...')
    selector_final = SelectKBest(score_func=chi2, k=MAX_FEATURES)
    X_final        = selector_final.fit_transform(vectores, targets)
    modelo_final   = SVC(kernel='linear', probability=True, random_state=42)
    modelo_final.fit(X_final, targets)

    metricas = {
        'clasificador':    'SVM (kernel lineal)',
        'features_method': f'chi2 (k={MAX_FEATURES})',
        'n_folds':         N_FOLDS,
        'n_docs':          n_docs,
        'n_features_orig': n_feat,
        'n_features_sel':  MAX_FEATURES,
        'clases':          clases,
        'accuracy_folds':  [round(a, 4) for a in accuracy_folds],
        'accuracy_mean':   round(acc_mean, 4),
        'accuracy_std':    round(acc_std, 4),
        'auc_por_clase':   {clases[i]: round(auc_mean[i], 4) for i in range(n_clases)},
        'confusion_matrix': conf_matrix_total.tolist(),
        'elapsed_s':       round(elapsed, 1),
    }

    with open(METRICAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {'modelo': modelo_final, 'selector': selector_final, 'label_encoder': label_encoder},
        MODELO_FILE,
    )

    print(f'Modelo guardado:   {MODELO_FILE}')
    print(f'Métricas guardadas: {METRICAS_FILE}')
