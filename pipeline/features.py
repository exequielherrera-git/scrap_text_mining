# -*- coding: utf-8 -*-
"""
Transforma los HTML descargados por scrap_pagina12.py en un dataset de
entrenamiento para scikit-learn.

Técnica de extracción: Técnica 2 (marcadores en texto plano).
  - Página 12 usa el framework Arc XP / Fusion: el artículo completo viaja
    como JSON embebido en un <script> con el marcador 'Fusion.globalContent='.
  - El JSON contiene 'content_elements' con los párrafos y 'display_date'
    con la fecha de publicación.
  - Fallback: si el marcador no se encuentra, se extrae el texto visible de
    div.article-wrapper (título + bajada, siempre presentes en el HTML).

Estructura esperada de directorios:
    data/raw/
        economia/   ← un .html por artículo
        sociedad/
        el-mundo/
        el-pais/

Salida:
    data/artifacts/vectores.joblib   - matriz dispersa de BoW (documentos × features)
    data/artifacts/targets.joblib    - etiquetas de categoría por documento
    data/artifacts/features.joblib   - nombre de cada columna del vocabulario
    data/artifacts/fechas.joblib     - fecha de publicación (ISO 8601, vacía si no se encontró)
    data/artifacts/textos.joblib     - textos crudos (para BERT u otros embeddings)
"""
import json
import sys
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import joblib
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).parent))  # permite importar tokenizadores.py del mismo dir
from tokenizadores import tokenizador

# Algunos content_elements de Fusion contienen solo URLs como 'content';
# BeautifulSoup lo advierte pero es comportamiento esperado aquí.
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)

# ── Rutas ──────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent

DIR_BASE_CATEGORIAS = _ROOT / 'data' / 'raw'

# Preferir stopwords sin acentos: mayor cobertura porque el tokenizador
# puede devolver tokens con o sin tilde según el texto fuente.
_STOPWORDS_SIN_ACENTOS = _ROOT / 'resources' / 'stopwords_es_sin_acentos.txt'
_STOPWORDS_CON_ACENTOS = _ROOT / 'resources' / 'stopwords_es.txt'

# ── Archivos de salida ─────────────────────────────────────────────────────────
ARTIFACTS_DIR      = _ROOT / 'data' / 'artifacts'
VECTORS_FILE       = str(ARTIFACTS_DIR / 'vectores.joblib')
TARGETS_FILE       = str(ARTIFACTS_DIR / 'targets.joblib')
FEATURE_NAMES_FILE = str(ARTIFACTS_DIR / 'features.joblib')
FECHAS_FILE        = str(ARTIFACTS_DIR / 'fechas.joblib')
TEXTOS_FILE        = str(ARTIFACTS_DIR / 'textos.joblib')  # textos crudos, usados por BERT

# ── Parámetros del vectorizador BoW ───────────────────────────────────────────
MIN_DF:     int   = 3    # ignorar términos que aparecen en menos de 3 documentos
MAX_DF:     float = 0.8  # ignorar términos que aparecen en más del 80 % de docs
MIN_NGRAMS: int   = 1
MAX_NGRAMS: int   = 2    # incluir bigramas ("banco central", "derechos humanos", …)

# Artículos con menos de este número de caracteres se descartan: suelen ser
# artículos de solo foto, video o con cuerpo vacío en el JSON Fusion.
MIN_TEXTO_LEN = 200


# ── Extracción de contenido (Técnica 2: marcadores en texto plano) ─────────────

def _texto_de_elemento(el: dict) -> str:
    """Extrae texto plano de un content_element de Fusion según su tipo.

    Incluye: text, header (con 'level'), list.
    Ignora: image, gallery, raw_html, custom_embed (sin texto útil).
    """
    tipo = el.get('type')

    if tipo == 'text':
        raw = el.get('content', '')
        return BeautifulSoup(raw, 'html.parser').get_text(' ', strip=True) if raw else ''

    if tipo == 'header' and 'level' in el:
        # Solo subtítulos reales; headers sin 'level' son listas de relacionados
        raw = el.get('content', '')
        return BeautifulSoup(raw, 'html.parser').get_text(' ', strip=True) if raw else ''

    if tipo == 'list':
        return ' '.join(
            BeautifulSoup(item.get('content', ''), 'html.parser').get_text(' ', strip=True)
            for item in el.get('items', [])
            if item.get('type') == 'text' and item.get('content')
        )

    return ''


def _fusion_content(html_doc: str) -> dict | None:
    """
    Localiza el marcador 'Fusion.globalContent=' en el HTML y parsea
    el JSON que le sigue sin necesitar un marcador de fin.

    Retorna el dict del artículo, o None si el marcador no existe o el
    JSON no es un objeto.
    """
    MARCADOR = 'Fusion.globalContent='
    idx = html_doc.find(MARCADOR)
    if idx == -1:
        return None

    # raw_decode(s, idx) parsea a partir de la posición idx sin copiar
    # el string; más eficiente que html_doc[idx:] para documentos grandes.
    try:
        data, _ = json.JSONDecoder().raw_decode(html_doc, idx + len(MARCADOR))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _extraer_cuerpo(html_doc: str, data: dict | None) -> str | None:
    """Texto del artículo. Fuente primaria: JSON Fusion; fallback: HTML visible."""
    if data:
        candidato = ' '.join(
            t for el in data.get('content_elements', [])
            if (t := _texto_de_elemento(el))
        )
        if len(candidato) >= MIN_TEXTO_LEN:
            return candidato

    # Artículos de foto/video sin content_elements útiles: extraer del HTML
    soup = BeautifulSoup(html_doc, 'lxml')
    for tag in soup.select('script, style, nav, aside, header, footer'):
        tag.decompose()
    for selector in ('div.article-wrapper', 'article'):
        el = soup.select_one(selector)
        if el:
            candidato = el.get_text(separator=' ', strip=True)
            if len(candidato) >= MIN_TEXTO_LEN:
                return candidato

    return None


def _extraer_fecha(html_doc: str, data: dict | None) -> str:
    """Fecha de publicación ISO 8601. Fuente primaria: JSON Fusion; fallback: <time>."""
    if data:
        # display_date es la fecha editorial; first_publish_date la técnica
        fecha = data.get('display_date') or data.get('first_publish_date', '')
        if fecha:
            return fecha
    soup = BeautifulSoup(html_doc, 'lxml')
    time_el = soup.select_one('div.p12Date time, time[datetime]')
    if time_el:
        return time_el.get('datetime', time_el.get_text(strip=True))
    return ''


def extraer_texto_y_fecha(html_doc: str) -> tuple[str, str] | None:
    """Retorna (texto, fecha_iso) o None si el artículo no tiene texto suficiente."""
    data = _fusion_content(html_doc)
    texto = _extraer_cuerpo(html_doc, data)
    if texto is None:
        return None
    return texto, _extraer_fecha(html_doc, data)


# ── Lectura por directorio ─────────────────────────────────────────────────────

def leer_stopwords(filepath: Path) -> list[str]:
    lines = filepath.read_text(encoding='utf-8', errors='replace').splitlines()
    return [w.strip().lower() for w in lines if w.strip()]


def htmls_y_target(directorio: Path) -> tuple[list[str], list[str], list[str]]:
    """Lee los .html de un directorio de sección; retorna (textos, fechas, etiquetas)."""
    nombre_categoria = directorio.name
    textos: list[str] = []
    fechas: list[str] = []

    archivos_html = list(directorio.glob('*.html'))
    for archivo in archivos_html:
        html = archivo.read_text(encoding='utf-8', errors='replace')
        resultado = extraer_texto_y_fecha(html)
        if resultado is not None:
            texto, fecha = resultado
            textos.append(texto)
            fechas.append(fecha)
        else:
            print(f'  AVISO: sin texto extraíble en {archivo.name}')

    # Una etiqueta por documento (el nombre del subdirectorio es la clase)
    etiquetas = [nombre_categoria] * len(textos)
    print(f'  [{nombre_categoria}] {len(textos)}/{len(archivos_html)} artículos con texto')
    return textos, fechas, etiquetas


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    if not DIR_BASE_CATEGORIAS.is_dir():
        print(f'ERROR: no existe {DIR_BASE_CATEGORIAS}')
        print('Ejecutá primero scrap_pagina12.py para descargar los artículos.')
        raise SystemExit(1)

    # Cada subdirectorio de paginas12/ corresponde a una sección del diario
    categorias = sorted(d for d in DIR_BASE_CATEGORIAS.iterdir() if d.is_dir())
    if not categorias:
        print(f'ERROR: no hay subdirectorios en {DIR_BASE_CATEGORIAS}')
        raise SystemExit(1)

    print(f'Procesando {len(categorias)} categorías: {", ".join(d.name for d in categorias)}\n')

    todos_los_textos:  list[str] = []
    todos_los_targets: list[str] = []
    todas_las_fechas:  list[str] = []

    for categoria in categorias:
        textos, fechas, etiquetas = htmls_y_target(categoria)
        todos_los_textos.extend(textos)
        todos_los_targets.extend(etiquetas)
        todas_las_fechas.extend(fechas)

    print(f'\nTotal documentos: {len(todos_los_textos)}')
    fechas_validas = [f for f in todas_las_fechas if f]
    if fechas_validas:
        print(f'Rango de fechas: {min(fechas_validas)[:10]}  a  {max(fechas_validas)[:10]}')

    # Preferir stopwords sin acentos; si no existe, usar la versión con acentos
    stopwords_path = next(
        (p for p in (_STOPWORDS_SIN_ACENTOS, _STOPWORDS_CON_ACENTOS) if p.exists()),
        None,
    )
    mi_lista_stopwords = leer_stopwords(stopwords_path) if stopwords_path else []
    mi_tokenizer = tokenizador()

    print('\nVectorizando...')
    # CountVectorizer tokeniza, elimina stopwords y construye la matriz dispersa
    # documentos × vocabulario (representación Bag of Words).
    vectorizer = CountVectorizer(
        stop_words=mi_lista_stopwords,
        tokenizer=mi_tokenizer,
        lowercase=True,
        strip_accents='unicode',
        decode_error='ignore',
        ngram_range=(MIN_NGRAMS, MAX_NGRAMS),
        min_df=MIN_DF,
        max_df=MAX_DF,
    )
    todos_los_vectores = vectorizer.fit_transform(todos_los_textos)
    nombres_features = vectorizer.get_feature_names_out()

    # Persistir artefactos para que pipeline/model.py los cargue sin re-procesar los HTMLs.
    joblib.dump(todos_los_vectores, VECTORS_FILE)
    joblib.dump(todos_los_targets,  TARGETS_FILE)
    joblib.dump(todas_las_fechas,   FECHAS_FILE)
    joblib.dump(todos_los_textos,   TEXTOS_FILE)
    joblib.dump(nombres_features,   FEATURE_NAMES_FILE)

    print(f'\nDataset guardado:')
    print(f'  {VECTORS_FILE}  ({todos_los_vectores.shape[0]} docs × {todos_los_vectores.shape[1]} features)')
    print(f'  {TARGETS_FILE}')
    print(f'  {FEATURE_NAMES_FILE}')
    print(f'  {FECHAS_FILE}')
    print(f'  {TEXTOS_FILE}  (textos crudos para BERT)')
