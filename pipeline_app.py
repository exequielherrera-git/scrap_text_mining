# -*- coding: utf-8 -*-
"""
Pipeline Dashboard — Web Mining TP1 2026

Orquesta scrap_pagina12.py y de_html_a_tabla.py con un frontend web
en tiempo real.

Requiere:
    pip install flask

Uso:
    python pipeline_app.py
    Abrir http://localhost:5000 en el navegador
"""

import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import joblib
from flask import Flask, Response, jsonify, request, send_file, stream_with_context

app = Flask(__name__)

BASE_DIR      = Path(__file__).parent
PAGINAS_DIR   = BASE_DIR / 'data' / 'raw'
ARTIFACTS_DIR = BASE_DIR / 'data' / 'artifacts'
SECTIONS      = ['economia', 'el-pais', 'el-mundo', 'sociedad']

# ── Estado global ──────────────────────────────────────────────────────────────

_scraper_proc: subprocess.Popen | None = None
_scraper_lock  = threading.Lock()
_scraper_q: queue.Queue = queue.Queue(maxsize=1000)
_scraper_stop  = threading.Event()

_transform_proc: subprocess.Popen | None = None
_transform_lock = threading.Lock()
_transform_q: queue.Queue = queue.Queue(maxsize=1000)

_CLASSIFIERS   = ('svm', 'w2v')
_MODEL_SCRIPTS = {'svm': 'model.py', 'w2v': 'model_w2v.py'}
_METRICS_FILES = {'svm': 'metricas_svm.json', 'w2v': 'metricas_w2v.json'}

_model_procs:  dict[str, subprocess.Popen | None] = {c: None for c in _CLASSIFIERS}
_model_locks:  dict[str, threading.Lock]          = {c: threading.Lock() for c in _CLASSIFIERS}
_model_queues: dict[str, queue.Queue]             = {c: queue.Queue(maxsize=1000) for c in _CLASSIFIERS}

# Env compartido por todos los subprocesos del pipeline
_SUBPROCESS_ENV: dict = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _count_files() -> dict:
    return {
        s: len(list((PAGINAS_DIR / s).glob('*.html')))
        if (PAGINAS_DIR / s).exists() else 0
        for s in SECTIONS
    }


def _enqueue(q: queue.Queue, event: dict) -> None:
    try:
        q.put_nowait(event)
    except queue.Full:
        pass


SPIDER_NAME = 'pagina12_sitemap'

# Substrings case-sensitive y case-insensitive que indican una línea útil del log
_SCRAPY_INCLUDE = {
    f'[{SPIDER_NAME}]', 'Crawled (', '[omitido]', '[fuera-de-seccion]',
    'Redirecting', 'Retrying', 'Spider opened', 'Closing spider',
    'Spider closed', 'Enabled extensions',
}
_SCRAPY_INCLUDE_UPPER = {'ERROR', 'WARNING'}


def _scrapy_line_relevante(line: str) -> bool:
    """Retorna True para líneas de scrapy.log que merecen mostrarse en el dashboard."""
    return (
        any(k in line for k in _SCRAPY_INCLUDE)
        or any(k in line.upper() for k in _SCRAPY_INCLUDE_UPPER)
    )


def _tail_scrapy_log(log_path: Path, q: queue.Queue, stop: threading.Event) -> None:
    """
    Tail de scrapy.log en tiempo real, filtrado a eventos relevantes.

    Scrapy escribe a este archivo desde el proceso hijo (multiprocessing).
    Contiene requests HTTP, logs del spider, y métricas finales.
    """
    _enqueue(q, {'type': 'log', 'cls': 'default',
                 'msg': f'Esperando scrapy.log en {log_path} ...'})

    deadline = time.time() + 45
    while not log_path.exists() and not stop.is_set() and time.time() < deadline:
        time.sleep(0.5)

    if not log_path.exists():
        _enqueue(q, {'type': 'log', 'cls': 'aviso',
                     'msg': 'scrapy.log no apareció — verificá que el spider arrancó correctamente'})
        _enqueue(q, {'type': 'done'})
        return

    _enqueue(q, {'type': 'log', 'cls': 'ok', 'msg': 'scrapy.log detectado — leyendo eventos...'})

    with open(log_path, encoding='utf-8', errors='replace') as f:
        f.seek(0, 2)                     # ir al final del contenido pre-existente
        while not stop.is_set():
            line = f.readline()
            if line:
                line = line.rstrip()
                if _scrapy_line_relevante(line):
                    display = line[20:] if len(line) > 20 and line[4] == '-' else line  # strip timestamp
                    _enqueue(q, {'type': 'log', 'msg': display})
            else:
                time.sleep(0.15)

    # Flush final
    for line in f:
        line = line.rstrip()
        if line and _scrapy_line_relevante(line):
            display = line[20:] if len(line) > 20 and line[4] == '-' else line
            _enqueue(q, {'type': 'log', 'msg': display})

    _enqueue(q, {'type': 'done'})


def _capture_stdout(proc: subprocess.Popen, q: queue.Queue) -> None:
    """Lee stdout del proceso línea a línea y lo encola."""
    try:
        for raw in proc.stdout:
            line = raw.decode('utf-8', errors='replace').rstrip()
            if line:
                _enqueue(q, {'type': 'log', 'msg': line})
    except Exception as exc:
        _enqueue(q, {'type': 'log', 'cls': 'error', 'msg': f'Error leyendo stdout: {exc}'})
    finally:
        _enqueue(q, {'type': 'done'})


def _sse(q: queue.Queue) -> Response:
    """Genera una Response SSE que consume la queue hasta recibir 'done'."""
    def generate():
        while True:
            try:
                event = q.get(timeout=20)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get('type') == 'done':
                    break
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


def _clear_queue(q: queue.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


# ── API: Scraper ───────────────────────────────────────────────────────────────

@app.route('/api/scraper/start', methods=['POST'])
def scraper_start():
    global _scraper_proc, _scraper_stop

    body  = request.get_json(silent=True) or {}
    start = body.get('start', '')
    end   = body.get('end', '')

    with _scraper_lock:
        if _scraper_proc and _scraper_proc.poll() is None:
            return jsonify({'error': 'El scraper ya está en ejecución'}), 409

        cmd = [sys.executable, '-u', str(BASE_DIR / 'pipeline' / 'scraper.py')]
        if start:
            cmd += ['--start', start]
        if end:
            cmd += ['--end', end]

        _scraper_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,   # Scrapy stderr va al proceso hijo (multiprocessing)
            cwd=str(BASE_DIR),
            env=_SUBPROCESS_ENV,
        )

        _scraper_stop = threading.Event()
        _clear_queue(_scraper_q)

        # Borrar scrapy.log anterior para empezar limpio en cada ejecución
        scrapy_log = PAGINAS_DIR / 'scrapy.log'
        try:
            scrapy_log.unlink(missing_ok=True)
        except OSError:
            pass

        threading.Thread(
            target=_tail_scrapy_log,
            args=(scrapy_log, _scraper_q, _scraper_stop),
            daemon=True,
        ).start()

        # Hilo que detecta cuando el proceso termina y dispara el stop del tail
        def _watch():
            _scraper_proc.wait()
            time.sleep(1.5)          # dar tiempo al tail para el último flush
            _scraper_stop.set()

        threading.Thread(target=_watch, daemon=True).start()

    return jsonify({'status': 'started', 'pid': _scraper_proc.pid})


@app.route('/api/scraper/stop', methods=['POST'])
def scraper_stop():
    with _scraper_lock:
        if _scraper_proc and _scraper_proc.poll() is None:
            _scraper_proc.terminate()
            _scraper_stop.set()
            return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})


@app.route('/api/scraper/status')
def scraper_status():
    with _scraper_lock:
        running = _scraper_proc is not None and _scraper_proc.poll() is None
    return jsonify({'running': running, 'counts': _count_files()})


@app.route('/api/scraper/stream')
def scraper_stream():
    return _sse(_scraper_q)


# ── API: Transform ─────────────────────────────────────────────────────────────

@app.route('/api/transform/start', methods=['POST'])
def transform_start():
    global _transform_proc

    with _transform_lock:
        if _transform_proc and _transform_proc.poll() is None:
            return jsonify({'error': 'La vectorización ya está en ejecución'}), 409

        _transform_proc = subprocess.Popen(
            [sys.executable, '-u', str(BASE_DIR / 'pipeline' / 'features.py')],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env=_SUBPROCESS_ENV,
        )
        _clear_queue(_transform_q)

        threading.Thread(
            target=_capture_stdout,
            args=(_transform_proc, _transform_q),
            daemon=True,
        ).start()

    return jsonify({'status': 'started'})


@app.route('/api/transform/status')
def transform_status():
    with _transform_lock:
        running = _transform_proc is not None and _transform_proc.poll() is None
    done = all(
        (ARTIFACTS_DIR / f).exists()
        for f in ['vectores.joblib', 'targets.joblib', 'features.joblib']
    )
    return jsonify({'running': running, 'done': done})


@app.route('/api/transform/stream')
def transform_stream():
    return _sse(_transform_q)


# ── API: Model ────────────────────────────────────────────────────────────────

@app.route('/api/model/start', methods=['POST'])
def model_start():
    body = request.get_json(silent=True) or {}
    clf  = body.get('classifier', 'svm')

    if clf not in _CLASSIFIERS:
        return jsonify({'error': f'Clasificador desconocido: "{clf}"'}), 400

    with _model_locks[clf]:
        if _model_procs[clf] and _model_procs[clf].poll() is None:
            return jsonify({'error': f'El entrenamiento {clf.upper()} ya está en ejecución'}), 409

        script = BASE_DIR / 'pipeline' / _MODEL_SCRIPTS[clf]
        _model_procs[clf] = subprocess.Popen(
            [sys.executable, '-u', str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(BASE_DIR),
            env=_SUBPROCESS_ENV,
        )
        _clear_queue(_model_queues[clf])

        threading.Thread(
            target=_capture_stdout,
            args=(_model_procs[clf], _model_queues[clf]),
            daemon=True,
        ).start()

    return jsonify({'status': 'started', 'classifier': clf})


@app.route('/api/model/stop', methods=['POST'])
def model_stop():
    body = request.get_json(silent=True) or {}
    clf  = body.get('classifier', 'svm')

    if clf not in _CLASSIFIERS:
        return jsonify({'error': f'Clasificador desconocido: "{clf}"'}), 400

    with _model_locks[clf]:
        if _model_procs[clf] and _model_procs[clf].poll() is None:
            _model_procs[clf].terminate()
            return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_running'})


@app.route('/api/model/status')
def model_status():
    result = {}
    for clf in _CLASSIFIERS:
        with _model_locks[clf]:
            running = _model_procs[clf] is not None and _model_procs[clf].poll() is None
        done = (ARTIFACTS_DIR / _METRICS_FILES[clf]).exists()
        result[clf] = {'running': running, 'done': done}
    return jsonify(result)


@app.route('/api/model/stream')
def model_stream():
    clf = request.args.get('clf', 'svm')
    if clf not in _CLASSIFIERS:
        return jsonify({'error': 'Clasificador desconocido'}), 400
    return _sse(_model_queues[clf])


@app.route('/api/model/metrics')
def model_metrics():
    clf = request.args.get('clf', 'svm')
    if clf not in _CLASSIFIERS:
        return jsonify({'error': 'Clasificador desconocido'}), 400
    path = ARTIFACTS_DIR / _METRICS_FILES[clf]
    if not path.exists():
        return jsonify({'error': f'No hay métricas para {clf}. Entrenás el modelo primero.'}), 404
    with open(path, encoding='utf-8') as f:
        return jsonify(json.load(f))


# ── API: KPIs ──────────────────────────────────────────────────────────────────

@app.route('/api/kpis')
def kpis():
    return jsonify(_count_files())


# ── API: Preview de datos ──────────────────────────────────────────────────────

@app.route('/api/data/preview')
def data_preview():
    needed = ['vectores.joblib', 'targets.joblib', 'features.joblib',
              'textos.joblib', 'fechas.joblib']
    missing = [f for f in needed if not (ARTIFACTS_DIR / f).exists()]
    if missing:
        return jsonify({'error': f'Archivos no encontrados: {missing}. '
                                 f'Ejecutá primero la vectorización.'}), 404

    targets  = joblib.load(ARTIFACTS_DIR / 'targets.joblib')
    features = list(joblib.load(ARTIFACTS_DIR / 'features.joblib'))
    textos   = joblib.load(ARTIFACTS_DIR / 'textos.joblib')
    fechas   = joblib.load(ARTIFACTS_DIR / 'fechas.joblib')
    vectores = joblib.load(ARTIFACTS_DIR / 'vectores.joblib')

    n_docs, n_feat = vectores.shape
    n_cols = min(10, n_feat)
    n_rows = min(10, n_docs)
    bow_sub   = vectores[:n_rows, :n_cols].toarray().tolist()
    top_feats = features[:n_cols]

    return jsonify({
        'summary': {
            'n_docs':      n_docs,
            'n_features':  n_feat,
            'por_seccion': {s: targets.count(s) for s in SECTIONS},
        },
        'tabs': [
            {
                'id':    'vectores',
                'label': 'vectores.joblib',
                'role':  '★ Alimenta el modelo (X)',
                'desc':  (f'Matriz dispersa {n_docs} docs × {n_feat} features. '
                          f'Cada celda = cantidad de veces que aparece esa palabra en ese artículo. '
                          f'Se muestran las primeras {n_rows} filas y {n_cols} features.'),
                'columns': ['#', 'Sección'] + top_feats,
                'rows': [
                    [i, targets[i]] + [int(v) for v in bow_sub[i]]
                    for i in range(n_rows)
                ],
            },
            {
                'id':    'targets',
                'label': 'targets.joblib',
                'role':  '★ Etiquetas del modelo (y)',
                'desc':  (f'Lista de {n_docs} strings. Una etiqueta por documento: '
                          f'el nombre de la sección del artículo. '
                          f'Es la variable que el clasificador aprende a predecir.'),
                'columns': ['#', 'Sección (etiqueta)'],
                'rows':    [[i, targets[i]] for i in range(n_rows)],
            },
            {
                'id':    'features',
                'label': 'features.joblib',
                'role':  'Nombres de columnas de vectores.joblib',
                'desc':  (f'Array de {n_feat} strings: el vocabulario completo del corpus '
                          f'(palabras y bigramas). El índice de cada feature coincide '
                          f'con la columna correspondiente en vectores.joblib.'),
                'columns': ['Índice', 'Feature (palabra / bigrama)'],
                'rows':    [[i, features[i]] for i in range(min(10, n_feat))],
            },
            {
                'id':    'textos',
                'label': 'textos.joblib',
                'role':  'Texto crudo (para BERT / embeddings)',
                'desc':  (f'Lista de {n_docs} strings con el texto completo de cada artículo '
                          f'antes de vectorizar. No lo usa el clasificador BoW, '
                          f'pero es necesario para modelos que trabajan con texto completo '
                          f'(BERT, Word2Vec, TF-IDF, etc.).'),
                'columns': ['#', 'Sección', 'Fecha', 'Chars', 'Texto (primeros 150 chars)'],
                'rows': [
                    [i, targets[i], (fechas[i] or '')[:10],
                     len(textos[i]), textos[i][:150]]
                    for i in range(n_rows)
                ],
                'fullTexts': [textos[i] for i in range(n_rows)],
            },
            {
                'id':    'fechas',
                'label': 'fechas.joblib',
                'desc':  (f'Lista de {n_docs} strings ISO 8601 con la fecha de publicación '
                          f'de cada artículo. Mismo índice que targets y vectores. '
                          f'No entra al clasificador pero es útil para análisis temporal.'),
                'columns': ['#', 'Sección', 'Fecha de publicación'],
                'rows': [
                    [i, targets[i], fechas[i] or '(sin fecha)']
                    for i in range(n_rows)
                ],
            },
        ],
    })


# ── Frontend ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_file(BASE_DIR / 'pipeline_frontend.html')


# ── Gestión del puerto y cierre limpio ────────────────────────────────────────

PORT = 5000


def _free_port(port: int) -> None:
    """Si el puerto está en uso, mata el proceso que lo ocupa."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        if s.connect_ex(('127.0.0.1', port)) != 0:
            return  # puerto libre

    print(f'  Puerto {port} ocupado — buscando proceso...')
    try:
        if sys.platform == 'win32':
            out = subprocess.run(
                ['netstat', '-ano'], capture_output=True, text=True
            ).stdout
            for line in out.splitlines():
                if f':{port} ' in line and 'LISTENING' in line:
                    pid = int(line.split()[-1])
                    if pid and pid != os.getpid():
                        subprocess.run(
                            ['taskkill', '/F', '/PID', str(pid)],
                            capture_output=True,
                        )
                        print(f'  PID {pid} terminado — puerto {port} liberado.')
                        time.sleep(0.6)
                        break
        else:
            out = subprocess.run(
                ['fuser', f'{port}/tcp'], capture_output=True, text=True
            ).stdout
            for pid_str in out.split():
                pid = int(pid_str)
                if pid != os.getpid():
                    os.kill(pid, signal.SIGKILL)
                    print(f'  PID {pid} terminado — puerto {port} liberado.')
                    time.sleep(0.6)
    except Exception as exc:
        print(f'  Advertencia al liberar puerto: {exc}')


def _shutdown(sig, frame) -> None:
    """Ctrl+C / SIGTERM: termina subprocesos y cierra limpiamente."""
    print('\n  Cerrando Pipeline Dashboard...')
    with _scraper_lock:
        if _scraper_proc and _scraper_proc.poll() is None:
            print('  Deteniendo scraper...')
            _scraper_proc.terminate()
            try:
                _scraper_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _scraper_proc.kill()
    with _transform_lock:
        if _transform_proc and _transform_proc.poll() is None:
            print('  Deteniendo vectorizacion...')
            _transform_proc.terminate()
            try:
                _transform_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _transform_proc.kill()
    for clf in _CLASSIFIERS:
        with _model_locks[clf]:
            if _model_procs[clf] and _model_procs[clf].poll() is None:
                print(f'  Deteniendo entrenamiento {clf.upper()}...')
                _model_procs[clf].terminate()
                try:
                    _model_procs[clf].wait(timeout=3)
                except subprocess.TimeoutExpired:
                    _model_procs[clf].kill()
    print(f'  Puerto {PORT} liberado. Hasta luego.')
    os._exit(0)


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    _free_port(PORT)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print()
    print(f'  Pipeline Dashboard  ->  http://localhost:{PORT}')
    print('  Ctrl+C para cerrar.')
    print()
    app.run(debug=False, threaded=True, port=PORT, use_reloader=False)
