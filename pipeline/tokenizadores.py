"""Tokenizers y stemmers para español. Renombrado de tokenizers.py para evitar
colisión con el paquete 'tokenizers' de HuggingFace."""
import re
from typing import Callable, Pattern

from nltk.stem.snowball import SnowballStemmer

# Instancia global: SnowballStemmer carga datos al construirse, no en cada llamada
_stemmer = SnowballStemmer('spanish')

_DEFAULT_TOKEN_RE: str = (
    r'[a-zA-ZâáàãõáêéíóôõúüÁÉÍÓÚñÑçÇ]'
    r'[0-9a-zA-ZâáàãõáêéíóôõúüÁÉÍÓÚñÑçÇ]+'
)


def stem(tokens: list[str]) -> list[str]:
    """Aplica stemming en español a cada token."""
    return [_stemmer.stem(w.lower()) for w in tokens]


def tokenizador(token_regex: str | Pattern | None = None) -> Callable[[str], list[str]]:
    """Retorna una función que tokeniza texto según la regex dada (o la default)."""
    pattern = re.compile(token_regex if token_regex is not None else _DEFAULT_TOKEN_RE)
    return pattern.findall


def tokenizador_con_stemming(token_regex: str | Pattern | None = None) -> Callable[[str], list[str]]:
    """Como tokenizador(), pero aplica stemming a cada token resultante."""
    tok = tokenizador(token_regex)
    return lambda doc: stem(tok(doc))
