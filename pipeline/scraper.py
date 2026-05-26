# -*- coding: utf-8 -*-
"""
Descarga artículos de cuatro secciones de Página 12 usando sitemaps XML diarios
y los guarda como archivos HTML individuales, organizados por sección.

Estrategia de crawling (sitemaps en vez de paginación):
  - Navega /arc/outboundfeeds/sitemap/YYYY-MM-DD/ por cada día del rango.
    Los sitemaps son estáticos y no dependen de scroll infinito ni JS.
  - La sección real de cada artículo se lee del <meta name="p12:section">
    del HTML descargado (más preciso que inferirla por la URL de índice).
  - Idempotencia: si el archivo ya existe, se omite sin re-descargarlo.

Salida:
    data/raw/
        economia/       ← {YYYY-MM-DD}-{slug}.html por artículo
        sociedad/
        el-mundo/
        el-pais/
        progreso.log    ← registro de avance (fecha/hora + nombre de archivo)
        dashboard.html  ← estado en tiempo real; abrir en el navegador

Uso:
    python pipeline/scraper.py                        # últimos 3 días (por defecto)
    python pipeline/scraper.py --start 2026-01-01     # desde fecha específica
    python pipeline/scraper.py --start 2026-01-01 --end 2026-03-31
"""

import argparse
import logging
import multiprocessing
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

import scrapy
from bs4 import BeautifulSoup
from scrapy.crawler import CrawlerProcess

# ── Configuración ──────────────────────────────────────────────────────────────

TARGET_SECTIONS = {'economia', 'el-pais', 'el-mundo', 'sociedad'}

SITEMAP_URL = (
    'https://www.pagina12.com.ar/arc/outboundfeeds/sitemap/{date}/?outputType=xml'
)

NEWS_URL_RE = re.compile(
    r'https?://(?:www\.)?pagina12\.com\.ar'
    r'/(?P<year>\d{4})/(?P<month>\d{2})/(?P<day>\d{2})/(?P<slug>[a-z0-9][a-z0-9-]*)/?$',
    re.IGNORECASE,
)

DIAS_ATRAS_DEFAULT = 2   # últimos 3 días: anteayer, ayer y hoy


# ── Evento ─────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Evento:
    ts: str
    seccion: str
    url: str
    archivo: str
    estado: str


# ── Dashboard HTML ─────────────────────────────────────────────────────────────

class DashboardHTML:
    """
    Genera y actualiza un archivo HTML con el estado del scraping en tiempo real.

    En cada evento (guardado, omitido, error) regenera el HTML y lo escribe a
    disco. El navegador lo refresca automáticamente cada 3 s gracias a
    <meta http-equiv="refresh">.
    """

    _MAX_EVENTOS = 300

    def __init__(self, filepath: Path, contadores: dict, rango: str) -> None:
        self.filepath    = filepath
        self.contadores  = contadores   # referencia viva al dict del spider
        self._rango      = rango
        self._eventos: list[Evento] = []
        self._inicio     = datetime.now()
        self._finalizado = False
        self._escribir()

    def registrar(self, seccion: str, url: str, archivo: str, estado: str) -> None:
        self._eventos.append(Evento(
            ts=datetime.now().strftime('%H:%M:%S'),
            seccion=seccion,
            url=url,
            archivo=archivo,
            estado=estado,
        ))
        if len(self._eventos) > self._MAX_EVENTOS:
            self._eventos.pop(0)
        self._escribir()

    def finalizar(self) -> None:
        self._finalizado = True
        self._escribir()

    def _escribir(self) -> None:
        try:
            self.filepath.write_text(self._render(), encoding='utf-8')
        except OSError:
            pass   # no bloquear el scraper por un error del dashboard

    def _fila_evento(self, ev: Evento) -> str:
        if ev.estado == 'guardado':
            css, icon = 'ok', '✓'
        elif ev.estado == 'omitido':
            css, icon = 'skip', '↩'
        else:
            css, icon = 'err', '✗'
        return (
            f'<tr class="{css}">'
            f'<td class="mono">{escape(ev.ts)}</td>'
            f'<td>{escape(ev.seccion)}</td>'
            f'<td class="center">{icon}</td>'
            f'<td>{escape(ev.estado)}</td>'
            f'<td class="mono small url">{escape(ev.url)}</td>'
            f'<td class="mono small">{escape(ev.archivo)}</td>'
            f'</tr>'
        )

    def _render(self) -> str:
        ahora   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = str(datetime.now() - self._inicio).split('.')[0]
        total   = sum(self.contadores.values())

        estado_txt = 'Finalizado ✓' if self._finalizado else 'En curso…'
        estado_cls = 'done'         if self._finalizado else 'running'

        filas_prog = ''.join(
            f'<tr><td>{escape(s)}</td><td class="r">{self.contadores.get(s, 0)}</td></tr>'
            for s in sorted(TARGET_SECTIONS)
        )
        filas_ev = ''.join(self._fila_evento(ev) for ev in reversed(self._eventos))

        return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="3">
<title>Scraping Página 12 — {estado_txt}</title>
<style>
  * {{ box-sizing: border-box; }}
  body  {{ font-family: system-ui, -apple-system, sans-serif;
           margin: 0; padding: 1.5rem 2rem;
           background: #f0f2f5; color: #1a1a1a; }}
  h1    {{ font-size: 1.4rem; margin: 0 0 .3rem; }}
  h2    {{ font-size: .85rem; margin: 1.5rem 0 .4rem; color: #555;
           text-transform: uppercase; letter-spacing: .06em; }}
  .badge         {{ display: inline-block; padding: .15rem .7rem;
                    border-radius: 1rem; font-size: .82rem; font-weight: 600;
                    vertical-align: middle; }}
  .badge.running {{ background: #fff3cd; color: #856404; }}
  .badge.done    {{ background: #d1e7dd; color: #0a5c36; }}
  .meta  {{ color: #666; font-size: .85rem; margin-bottom: 1.2rem; }}
  table  {{ border-collapse: collapse; width: 100%; background: #fff;
             box-shadow: 0 1px 4px rgba(0,0,0,.08);
             border-radius: .5rem; overflow: hidden; margin-bottom: 1rem; }}
  th, td {{ padding: .5rem .85rem; text-align: left;
             border-bottom: 1px solid #f0f0f0; }}
  th     {{ background: #f7f7f7; font-size: .78rem; font-weight: 600;
             color: #666; text-transform: uppercase; letter-spacing: .05em; }}
  tr:last-child td {{ border-bottom: none; }}
  .r      {{ text-align: right; }}
  .center {{ text-align: center; }}
  .bar    {{ background: #e8e8e8; border-radius: .4rem;
              height: .85rem; width: 10rem; }}
  .fill   {{ background: #2ecc71; border-radius: .4rem; height: 100%; }}
  .ok   td {{ background: #f0fff4; }}
  .skip td {{ color: #aaa; }}
  .err  td {{ background: #fff5f5; color: #c0392b; }}
  .mono    {{ font-family: ui-monospace, "Cascadia Code", monospace;
               font-size: .82rem; }}
  .small   {{ font-size: .78rem; }}
  .url     {{ max-width: 32rem; overflow: hidden;
               text-overflow: ellipsis; white-space: nowrap; }}
</style>
</head>
<body>
<h1>Scraping Página 12 &nbsp;
    <span class="badge {estado_cls}">{estado_txt}</span></h1>
<p class="meta">
    Rango: {escape(self._rango)} &nbsp;·&nbsp;
    Actualizado: {ahora} &nbsp;·&nbsp;
    Tiempo: {elapsed} &nbsp;·&nbsp;
    Total guardados: <strong>{total}</strong>
</p>

<h2>Progreso por sección</h2>
<table>
  <thead>
    <tr>
      <th>Sección</th>
      <th class="r">Guardados</th>
    </tr>
  </thead>
  <tbody>{filas_prog}</tbody>
</table>

<h2>Registro de eventos — {len(self._eventos)} de {self._MAX_EVENTOS} máx.</h2>
<table>
  <thead>
    <tr>
      <th>Hora</th>
      <th>Sección</th>
      <th class="center">OK</th>
      <th>Estado</th>
      <th>URL scrapeada</th>
      <th>Archivo generado</th>
    </tr>
  </thead>
  <tbody>{filas_ev}</tbody>
</table>
</body>
</html>"""


# ── Spider ─────────────────────────────────────────────────────────────────────

class Pagina12Spider(scrapy.Spider):
    """
    Spider que descarga artículos de Página 12 mediante sitemaps XML diarios.

    Flujo:
        start           → un request por sitemap del día en el rango de fechas
        parse_sitemap   → extrae URLs de artículos del XML con re.findall
        parse_article   → lee <meta name="p12:section">, guarda el HTML
                          en paginas12/{seccion}/{YYYY-MM-DD}-{slug}.html
    """

    name             = 'pagina12_sitemap'
    allowed_domains  = ['www.pagina12.com.ar', 'pagina12.com.ar']

    custom_settings = {
        'USER_AGENT': (
            'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) '
            'AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'
        ),
        'LOG_ENABLED':  True,
        'LOG_LEVEL':    'DEBUG',   # DEBUG para ver cada request/response en scrapy.log
        'LOG_FILE':     str(Path(__file__).parent.parent / 'data' / 'raw' / 'scrapy.log'),
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_DELAY': 1.5,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'AUTOTHROTTLE_ENABLED':          True,
        'AUTOTHROTTLE_START_DELAY':      1.0,
        'AUTOTHROTTLE_MAX_DELAY':        10.0,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
        'RETRY_TIMES': 3,
        'HTTPERROR_ALLOWED_CODES': [403, 404],
        # Limpiar DOWNLOAD_HANDLERS evita heredar stubs de proyectos anteriores
        # que podían causar errores silenciosos en las descargas.
        'DOWNLOAD_HANDLERS': {},
    }

    def __init__(self, start_date: str, end_date: str,
                 save_pages_in_dir: str = '.', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_date = date.fromisoformat(start_date)
        self.end_date   = date.fromisoformat(end_date)
        self.basedir    = Path(save_pages_in_dir)
        self._articulos_guardados = {s: 0 for s in TARGET_SECTIONS}

        for section in TARGET_SECTIONS:
            (self.basedir / section).mkdir(parents=True, exist_ok=True)

        self._log = self._setup_progress_log(self.basedir / 'progreso.log')
        self._log.info('=' * 60)
        self._log.info('INICIO DEL SCRAPING')
        self._log.info(f'Rango       : {start_date} → {end_date}')
        self._log.info(f'Secciones   : {", ".join(sorted(TARGET_SECTIONS))}')
        self._log.info(f'Carpeta     : {self.basedir}')
        self._log.info('=' * 60)

        self._dashboard = DashboardHTML(
            filepath=self.basedir / 'dashboard.html',
            contadores=self._articulos_guardados,
            rango=f'{start_date} → {end_date}',
        )

    @staticmethod
    def _setup_progress_log(log_path: Path) -> logging.Logger:
        log = logging.getLogger('progreso_pagina12')
        log.setLevel(logging.INFO)
        if not log.handlers:  # evita duplicar handlers si Scrapy re-instancia el spider
            handler = logging.FileHandler(log_path, encoding='utf-8')
            handler.setFormatter(
                logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            )
            log.addHandler(handler)
        return log

    # ── Arranque ───────────────────────────────────────────────────────────────

    async def start(self):
        """Genera un request por sitemap diario en el rango configurado."""
        current = self.start_date
        while current <= self.end_date:
            yield scrapy.Request(
                url=SITEMAP_URL.format(date=current.isoformat()),
                callback=self.parse_sitemap,
                meta={'date': current.isoformat()},
            )
            current += timedelta(days=1)

    # ── Parseo del sitemap XML ─────────────────────────────────────────────────

    def parse_sitemap(self, response) -> None:
        """
        Extrae URLs de artículos del sitemap XML del día.

        El XML usa namespaces; re.findall sobre <loc> es más robusto que
        XPath con namespace y no requiere librerías adicionales.
        """
        dia   = response.meta['date']
        urls  = [
            loc for loc in re.findall(r'<loc>(https?://[^<]+)</loc>', response.text)
            if '/print/' not in loc and NEWS_URL_RE.search(loc)
        ]
        self.logger.info(f'[sitemap {dia}] {len(urls)} artículos encolados')
        for loc in urls:
            yield scrapy.Request(url=loc, callback=self.parse_article)

    # ── Parseo y guardado de artículos ─────────────────────────────────────────

    def parse_article(self, response) -> None:
        """Guarda el HTML en basedir/{seccion}/{fecha}-{slug}.html.

        La sección se lee de <meta name="p12:section"> — más preciso que inferirla
        por URL, ya que un artículo puede aparecer en una sección pero pertenecer a otra.
        """
        soup = BeautifulSoup(response.text, 'html.parser')
        meta = soup.find('meta', attrs={'name': 'p12:section'})
        if not meta or not meta.get('content'):
            return

        section = meta['content'].strip().lower()
        if section not in TARGET_SECTIONS:
            self.logger.info(f'[fuera-de-seccion] {section} → {response.url}')
            return

        match = NEWS_URL_RE.search(response.url)
        if not match:
            return

        filename = (
            f"{match.group('year')}-{match.group('month')}-"
            f"{match.group('day')}-{match.group('slug')}.html"
        )
        dest_file = self.basedir / section / filename

        if dest_file.exists():
            self.logger.info(f'[omitido:{section}] ya existe → {filename}')
            self._dashboard.registrar(section, response.url, filename, 'omitido')
            return

        try:
            dest_file.write_text(response.text, encoding='utf-8')
            self._articulos_guardados[section] += 1
            n = self._articulos_guardados[section]
            self.logger.info(f'[{section}] #{n:03d} guardado → {filename}')
            self._log.info(f'[{section}] #{n:03d} | {filename}')
            self._dashboard.registrar(section, response.url, filename, 'guardado')
        except OSError as exc:
            self.logger.error(f'Error guardando {dest_file}: {exc}')
            self._log.error(f'[{section}] ERROR al guardar {filename} — {exc}')
            self._dashboard.registrar(section, response.url, filename, f'error: {exc}')

    # ── Resumen al cerrar el spider ────────────────────────────────────────────

    def closed(self, reason: str) -> None:
        total = sum(self._articulos_guardados.values())
        self._log.info('=' * 60)
        self._log.info(f'FIN DEL SCRAPING — motivo: {reason}')
        for seccion, cantidad in sorted(self._articulos_guardados.items()):
            self._log.info(f'  {seccion:<12}: {cantidad:>3} artículos guardados')
        self._log.info(f'  {"TOTAL":<12}: {total:>3} artículos guardados')
        self._log.info('=' * 60)
        self._dashboard.finalizar()
        print(f'\nScraping finalizado. Total: {total} artículos guardados.')


# ── Punto de entrada ──────────────────────────────────────────────────────────

def _start_crawler(save_pages_in_dir: Path, start_date: str, end_date: str) -> None:
    """Lanza el spider en un proceso hijo para aislar el reactor de Twisted.

    El reactor de Twisted no puede reiniciarse una vez detenido en el mismo
    proceso (ReactorNotRestartable). El proceso hijo aísla el reactor y permite
    llamar a este módulo repetidamente (p.ej. desde pipeline_runner.py).
    """
    process = CrawlerProcess()
    process.crawl(
        Pagina12Spider,
        start_date=start_date,
        end_date=end_date,
        save_pages_in_dir=str(save_pages_in_dir),
    )
    process.start()


def main() -> None:
    today         = date.today().isoformat()
    default_start = (date.today() - timedelta(days=DIAS_ATRAS_DEFAULT)).isoformat()

    parser = argparse.ArgumentParser(
        description='Scraper Página 12 — sitemaps diarios (Web Mining)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--start', default=default_start, metavar='YYYY-MM-DD',
        help='Fecha de inicio del rango a descargar',
    )
    parser.add_argument(
        '--end', default=today, metavar='YYYY-MM-DD',
        help='Fecha de fin del rango (inclusive)',
    )
    args = parser.parse_args()

    dir_salida = Path(__file__).parent.parent / 'data' / 'raw'
    dir_salida.mkdir(exist_ok=True)

    print()
    print(f'  Descargando artículos en : {dir_salida}')
    print(f'  Rango                    : {args.start}  →  {args.end}')
    print(f'  Secciones                : {", ".join(sorted(TARGET_SECTIONS))}')
    print(f'  Dashboard (tiempo real)  : {dir_salida / "dashboard.html"}')
    print()

    p = multiprocessing.Process(
        target=_start_crawler,
        args=(dir_salida, args.start, args.end),
    )
    p.start()
    p.join()
    print('Scraping finalizado.')


if __name__ == '__main__':
    main()
