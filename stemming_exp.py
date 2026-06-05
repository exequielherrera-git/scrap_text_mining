# -*- coding: utf-8 -*-
"""
Experimento de stemming: compara BoW+chi2+SVM con y sin SnowballStemmer.
"""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import sys

sys.path.insert(0, str(Path(__file__).parent / 'pipeline'))
from tokenizadores import tokenizador, tokenizador_con_stemming

ROOT = Path(__file__).parent
ARTIFACTS_DIR = ROOT / 'data' / 'artifacts'
RESULTS_FILE = ARTIFACTS_DIR / 'stemming_experiment.json'

stopwords_path = next(
    (p for p in (
        ROOT / 'resources' / 'stopwords_es_sin_acentos.txt',
        ROOT / 'resources' / 'stopwords_es.txt',
    ) if p.exists()),
    None,
)
stopwords = stopwords_path.read_text(encoding='utf-8', errors='replace').splitlines() if stopwords_path else []
stopwords = [w.strip().lower() for w in stopwords if w.strip()]

textos = joblib.load(ARTIFACTS_DIR / 'textos.joblib')
nombres_targets = joblib.load(ARTIFACTS_DIR / 'targets.joblib')

le = LabelEncoder()
targets = le.fit_transform(nombres_targets)
n_docs = len(textos)
print(f"Corpus: {n_docs} documentos")

N_FOLDS = 5
MAX_FEATURES = 150
kf = StratifiedKFold(n_splits=N_FOLDS, random_state=42, shuffle=True)

def run_cv(tok_fn, label):
    print(f"\n--- {label} ---")
    vec = CountVectorizer(
        stop_words=stopwords,
        tokenizer=tok_fn,
        lowercase=True,
        strip_accents='unicode',
        decode_error='ignore',
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.8,
    )
    X = vec.fit_transform(textos)
    n_feat = X.shape[1]
    print(f"  Features: {n_feat}")

    acc_list, f1_list = [], []
    clf = SVC(kernel='linear', probability=True, random_state=42)

    for fold_n, (tr, te) in enumerate(kf.split(X, targets), 1):
        sel = SelectKBest(score_func=chi2, k=MAX_FEATURES)
        Xtr = sel.fit_transform(X[tr], targets[tr])
        Xte = sel.transform(X[te])
        clf.fit(Xtr, targets[tr])
        preds = clf.predict(Xte)
        acc = accuracy_score(targets[te], preds)
        f1 = f1_score(targets[te], preds, average='macro')
        acc_list.append(acc)
        f1_list.append(f1)
        print(f"  Fold {fold_n}: acc={acc:.4f}  f1={f1:.4f}")

    acc_mean = float(np.mean(acc_list))
    acc_std = float(np.std(acc_list))
    f1_mean = float(np.mean(f1_list))
    f1_std = float(np.std(f1_list))
    print(f"  Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  F1 macro: {f1_mean:.4f} ± {f1_std:.4f}")
    return {
        'n_features': n_feat,
        'accuracy_folds': [round(a, 4) for a in acc_list],
        'accuracy_mean': round(acc_mean, 4),
        'accuracy_std': round(acc_std, 4),
        'f1_folds': [round(f, 4) for f in f1_list],
        'f1_mean': round(f1_mean, 4),
        'f1_std': round(f1_std, 4),
    }

sin_stem = run_cv(tokenizador(), "Sin stemming")
con_stem = run_cv(tokenizador_con_stemming(), "Con stemming")

delta_acc = round(con_stem['accuracy_mean'] - sin_stem['accuracy_mean'], 4)
delta_f1 = round(con_stem['f1_mean'] - sin_stem['f1_mean'], 4)
print(f"\nDelta accuracy: {delta_acc:+.4f}")
print(f"Delta F1 macro: {delta_f1:+.4f}")

results = {
    'n_docs': n_docs,
    'n_folds': N_FOLDS,
    'chi2_k': MAX_FEATURES,
    'sin_stemming': sin_stem,
    'con_stemming': con_stem,
    'delta_accuracy': delta_acc,
    'delta_f1_macro': delta_f1,
}
with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nGuardado: {RESULTS_FILE}")
