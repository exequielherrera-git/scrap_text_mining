# -*- coding: utf-8 -*-
"""
Genera todas las imágenes para el informe TP1.
  images/distribucion_corpus.png
  images/confusion_matrix_cv.png
  images/metricas_comparacion.png
  images/f1_por_clase.png
  images/confusion_matrices_temporal.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / 'data' / 'artifacts'
IMAGES = ROOT / 'images'
IMAGES.mkdir(exist_ok=True)

# Cargar métricas
with open(ARTIFACTS / 'metricas_svm.json', encoding='utf-8') as f:
    bow = json.load(f)
with open(ARTIFACTS / 'metricas_w2v.json', encoding='utf-8') as f:
    lsa = json.load(f)
with open(ARTIFACTS / 'temporal_validation.json', encoding='utf-8') as f:
    temp = json.load(f)
with open(ARTIFACTS / 'stemming_experiment.json', encoding='utf-8') as f:
    stem = json.load(f)

CLASES = bow['clases']
CLASES_LABELS = ['Economía', 'El Mundo', 'El País', 'Sociedad']

# ── 1. Distribución del corpus ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Por sección
n_seccion = {
    'Economía': 580, 'El País': 1258, 'El Mundo': 486, 'Sociedad': 892
}
colors_sec = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0']
axes[0].bar(n_seccion.keys(), n_seccion.values(), color=colors_sec, edgecolor='white', linewidth=0.8)
axes[0].set_title('Artículos por sección', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Cantidad de artículos')
for i, (k, v) in enumerate(n_seccion.items()):
    axes[0].text(i, v + 10, str(v), ha='center', fontsize=10, fontweight='bold')
axes[0].set_ylim(0, max(n_seccion.values()) * 1.12)
axes[0].tick_params(axis='x', labelsize=10)

# Por mes
meses = ['Ene 2026', 'Feb 2026', 'Mar 2026', 'Abr 2026', 'May 2026']
n_mes = [331, 422, 186, 879, 1398]
colors_mes = ['#42A5F5'] * 3 + ['#EF5350'] * 2
bars = axes[1].bar(meses, n_mes, color=colors_mes, edgecolor='white', linewidth=0.8)
axes[1].set_title('Artículos por mes', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Cantidad de artículos')
for bar, v in zip(bars, n_mes):
    axes[1].text(bar.get_x() + bar.get_width()/2, v + 10, str(v),
                 ha='center', fontsize=10, fontweight='bold')
axes[1].set_ylim(0, max(n_mes) * 1.12)
axes[1].tick_params(axis='x', labelsize=9)
patch1 = mpatches.Patch(color='#42A5F5', label='Corpus inicial (ene-mar)')
patch2 = mpatches.Patch(color='#EF5350', label='Scraping adicional (abr-may)')
axes[1].legend(handles=[patch1, patch2], fontsize=9)

fig.suptitle(f'Corpus total: {bow["n_docs"]:,} artículos — Página 12 (ene-may 2026)',
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig(IMAGES / 'distribucion_corpus.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ distribucion_corpus.png")

# ── 2. Matrices de confusión CV ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
plt.subplots_adjust(wspace=0.08)

for ax, metricas, titulo in [
    (axes[0], bow, f'BoW + chi² + SVM\n(acc={bow["accuracy_mean"]:.2%} ±{bow["accuracy_std"]:.2%})'),
    (axes[1], lsa, f'TF-IDF + LSA + SVM\n(acc={lsa["accuracy_mean"]:.2%} ±{lsa["accuracy_std"]:.2%})'),
]:
    cm = np.array(metricas['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASES_LABELS)))
    ax.set_yticks(range(len(CLASES_LABELS)))
    ax.set_xticklabels(CLASES_LABELS, rotation=30, ha='right', fontsize=12)
    ax.set_yticklabels(CLASES_LABELS, fontsize=12)
    ax.set_xlabel('Predicción', fontsize=13)
    ax.set_ylabel('Real', fontsize=13)
    ax.set_title(titulo, fontsize=12, fontweight='bold', pad=12)
    thresh = 0.5
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                    ha='center', va='center', fontsize=11, color=color)

plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label='Proporción')
plt.suptitle('Matrices de confusión — Validación cruzada (5 folds)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(IMAGES / 'confusion_matrix_cv.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ confusion_matrix_cv.png")

# ── 3. Comparación de métricas ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

modelos = ['BoW + chi²\n+ SVM', 'TF-IDF + LSA\n+ SVM']
accs = [bow['accuracy_mean'], lsa['accuracy_mean']]
stds = [bow['accuracy_std'], lsa['accuracy_std']]
colors_mod = ['#1565C0', '#2E7D32']
bars = ax.bar(modelos, [a * 100 for a in accs], yerr=[s * 100 for s in stds],
              color=colors_mod, capsize=6, edgecolor='white', linewidth=0.8,
              error_kw={'elinewidth': 2, 'ecolor': 'gray'}, width=0.4)

for bar, acc, std in zip(bars, accs, stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            acc * 100 + std * 100 + 0.4,
            f'{acc:.2%}\n±{std:.2%}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylim(60, 100)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Comparación de accuracy (CV 5-fold)', fontsize=13, fontweight='bold')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
fig.savefig(IMAGES / 'metricas_comparacion.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ metricas_comparacion.png")

# ── 4. F1 por clase ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(CLASES_LABELS))
width = 0.35

# Calcular F1 por clase desde matriz de confusión
def f1_from_cm(cm_list):
    cm = np.array(cm_list)
    f1s = []
    for i in range(len(CLASES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1s.append(f1)
    return f1s

f1_bow = f1_from_cm(bow['confusion_matrix'])
f1_lsa = f1_from_cm(lsa['confusion_matrix'])

bars1 = ax.bar(x - width/2, [v * 100 for v in f1_bow], width, label='BoW + chi² + SVM',
               color='#1565C0', edgecolor='white')
bars2 = ax.bar(x + width/2, [v * 100 for v in f1_lsa], width, label='TF-IDF + LSA + SVM',
               color='#2E7D32', edgecolor='white')

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(CLASES_LABELS, fontsize=11)
ax.set_ylim(60, 105)
ax.set_ylabel('F1-score (%)', fontsize=12)
ax.set_title('F1-score por clase — Validación cruzada', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
plt.tight_layout()
fig.savefig(IMAGES / 'f1_por_clase.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ f1_por_clase.png")

# ── 5. Matrices de confusión temporales ──────────────────────────────────────
cortes = temp['cortes']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

titulos = [
    f'Corte 1: ene-mar → abr-may\n(acc={cortes[0]["accuracy"]:.2%})',
    f'Corte 2: ene-abr → may\n(acc={cortes[1]["accuracy"]:.2%})',
    f'Corte 3: 80% → 20%\n(acc={cortes[2]["accuracy"]:.2%})',
]
n_trains = [cortes[0]['n_train'], cortes[1]['n_train'], cortes[2]['n_train']]
n_tests  = [cortes[0]['n_test'],  cortes[1]['n_test'],  cortes[2]['n_test']]

for ax, corte, titulo, n_tr, n_te in zip(axes, cortes, titulos, n_trains, n_tests):
    cm = np.array(corte['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASES_LABELS)))
    ax.set_yticks(range(len(CLASES_LABELS)))
    ax.set_xticklabels(CLASES_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(CLASES_LABELS, fontsize=9)
    ax.set_xlabel('Predicción', fontsize=10)
    ax.set_ylabel('Real', fontsize=10)
    titulo_full = titulo + f'\ntrain={n_tr} | test={n_te}'
    ax.set_title(titulo_full, fontsize=10, fontweight='bold')
    thresh = 0.5
    for i in range(len(CLASES)):
        for j in range(len(CLASES)):
            color = 'white' if cm_norm[i, j] > thresh else 'black'
            ax.text(j, i, f'{cm_norm[i, j]:.2f}\n({cm[i, j]})',
                    ha='center', va='center', fontsize=8, color=color)

plt.suptitle('Matrices de confusión — Validación temporal (TF-IDF + LSA + SVM)', fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(IMAGES / 'confusion_matrices_temporal.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ confusion_matrices_temporal.png")

print("\nTodas las imágenes generadas en images/")
