# -*- coding: utf-8 -*-
"""
Word2Vec (MeanEmbeddingVectorizer) + SVM (kernel lineal).

Cada artículo se representa como el promedio de los vectores Word2Vec de sus
tokens (embeddings pre-entrenados, 300 dims). El SVM lineal clasifica sobre
esos vectores densos.

Requiere: resources/SBW-vectors-300-min5.bin
  Embeddings en español (Spanish Billion Word Corpus, DCC-UCh).
  Descargar de: https://github.com/dccuchile/spanish-word-embeddings

Artefactos de entrada (data/artifacts/):
    textos.joblib, targets.joblib

Artefactos de salida (data/artifacts/):
    modelo_embed.joblib   - modelo final entrenado sobre el dataset completo
    metricas_embed.json   - métricas de cross-validation para el dashboard
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import SVC

_ROOT         = Path(__file__).parent.parent
ARTIFACTS_DIR = _ROOT / 'data' / 'artifacts'
EMBEDDING_FILE = next(
    (p for p in (_ROOT / 'resources').glob('*.bin') if p.stat().st_size > 1_000_000),
    _ROOT / 'resources' / 'sbw_vectors.bin',
)

N_FOLDS      = 5
METRICAS_FILE = ARTIFACTS_DIR / 'metricas_embed.json'
MODELO_FILE   = ARTIFACTS_DIR / 'modelo_embed.joblib'

sys.path.insert(0, str(Path(__file__).parent))
from tokenizadores import tokenizador


# ── Loader nativo Word2Vec binario (sin gensim) ───────────────────────────────

class _W2VKeyedVectors:
    """Subconjunto mínimo de la API de gensim.KeyedVectors para MeanEmbeddingVectorizer."""
    def __init__(self, vocab: dict[str, np.ndarray], vector_size: int):
        self._vocab = vocab
        self.vector_size = vector_size

    def __len__(self) -> int:
        return len(self._vocab)

    def __contains__(self, word: str) -> bool:
        return word in self._vocab

    def get_vector(self, word: str) -> np.ndarray:
        return self._vocab[word]


def _load_word2vec_bin(path: Path) -> _W2VKeyedVectors:
    """
    Carga un archivo Word2Vec en formato binario (Google/SBW) sin gensim.

    Formato:
      - Primera línea ASCII: "<vocab_size> <vector_size>\\n"
      - Por cada palabra: bytes de la palabra + b' ' + float32 × vector_size (big-endian no, little-endian no, C float)
    """
    import struct
    vocab: dict[str, np.ndarray] = {}
    with open(path, 'rb') as f:
        # Leer header
        header = f.readline().decode('utf-8').strip()
        vocab_size, vector_size = (int(x) for x in header.split())
        float_bytes = vector_size * 4  # float32

        for _ in range(vocab_size):
            # Leer palabra (hasta espacio)
            word_chars = []
            while True:
                ch = f.read(1)
                if ch in (b' ', b'\t'):
                    break
                if ch == b'\n':
                    continue
                word_chars.append(ch)
            word = b''.join(word_chars).decode('utf-8', errors='replace')
            # Leer vector
            vec = np.frombuffer(f.read(float_bytes), dtype=np.float32).copy()
            vocab[word] = vec

    return _W2VKeyedVectors(vocab, vector_size)

from word2vec import MeanEmbeddingVectorizer

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
    if not EMBEDDING_FILE.exists():
        print(f'ERROR: no se encontró el archivo de embeddings:')
        print(f'  {EMBEDDING_FILE}')
        print('Descargarlo de: https://github.com/dccuchile/spanish-word-embeddings')
        sys.exit(1)

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

    print(f'Corpus: {len(textos)} docs  |  {n_clases} clases: {", ".join(clases)}')
    print(f'Representacion: Word2Vec MeanEmbedding (300 dims) — embeddings pre-entrenados')
    print(f'Clasificador: SVM kernel lineal  |  CV: {N_FOLDS} folds')
    print()

    print(f'Cargando embeddings Word2Vec: {EMBEDDING_FILE.name} ...')
    t_load = time.time()
    wv = _load_word2vec_bin(EMBEDDING_FILE)
    print(f'Embeddings cargados: {len(wv):,} palabras × {wv.vector_size} dims  ({time.time()-t_load:.1f}s)')
    print()

    stopwords  = _leer_stopwords()
    tokenize   = tokenizador()
    vectorizer = MeanEmbeddingVectorizer(wv, tokenize, stopwords)

    print('Vectorizando corpus (promedio de embeddings por artículo)...')
    t_vec = time.time()
    X = vectorizer.transform(textos)
    print(f'Vectorización lista: {X.shape[0]} docs × {X.shape[1]} dims  ({time.time()-t_vec:.1f}s)')
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
    elapsed    = time.time() - t_load
    acc_mean   = float(np.mean(accuracy_folds))
    acc_std    = float(np.std(accuracy_folds))
    auc_mean   = [float(np.mean([f[i] for f in auc_folds])) for i in range(n_clases)]

    print('== Resultados finales ==================================')
    print(f'Accuracy promedio: {acc_mean:.4f} ± {acc_std:.4f}')
    for i, clase in enumerate(clases):
        print(f'AUC {clase}: {auc_mean[i]:.4f}')
    print(f'Tiempo CV: {elapsed_cv:.1f}s  |  Total: {elapsed:.1f}s')
    print()

    print('Entrenando modelo final sobre dataset completo...')
    modelo_final = SVC(kernel='linear', probability=True, random_state=42)
    modelo_final.fit(X, targets)

    metricas = {
        'clasificador':    'Word2Vec + SVM (kernel lineal)',
        'features_method': f'MeanEmbeddingVectorizer ({wv.vector_size} dims)',
        'n_folds':         N_FOLDS,
        'n_docs':          len(textos),
        'n_features_orig': wv.vector_size,
        'n_features_sel':  wv.vector_size,
        'clases':          clases,
        'accuracy_folds':  [round(a, 4) for a in accuracy_folds],
        'accuracy_mean':   round(acc_mean, 4),
        'accuracy_std':    round(acc_std, 4),
        'auc_por_clase':   {clases[i]: round(auc_mean[i], 4) for i in range(n_clases)},
        'confusion_matrix': conf_matrix_total.tolist(),
        'elapsed_s':       round(elapsed, 1),
        'embedding_model': EMBEDDING_FILE.name,
    }

    with open(METRICAS_FILE, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    joblib.dump(
        {'modelo': modelo_final, 'vectorizer': vectorizer, 'label_encoder': label_encoder},
        MODELO_FILE,
    )

    print(f'Modelo guardado:    {MODELO_FILE}')
    print(f'Métricas guardadas: {METRICAS_FILE}')
