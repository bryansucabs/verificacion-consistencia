"""
Filtro previo al NLI: detecta si un mensaje contiene informacion factual concreta.

Si el mensaje no contiene hechos concretos (numeros, fechas, eventos,
lugares especificos) no vale la pena compararlo con el NLI porque
no puede actualizar ningun hecho guardado en Qdrant.

Esto reduce falsas contradicciones y acelera la carga porque DeBERTa
solo se invoca cuando el mensaje realmente podria contradecir algo.
"""

import re

# Numeros standalone (no seguidos de letras, para excluir "5K", "3D", "10x")
_NUM = r'\b\d+(?:\.\d+)?\b(?![A-Za-z])'

# Tiempos (25:50, 1:30:00)
_TIEMPO = r'\d+:\d+'

# Unidades de medida con numero
_UNIDADES = (
    r'\d+\s*'
    r'(km|mi|miles?|meters?|kg|lbs?|pounds?|calories?|'
    r'hours?|hrs?|minutes?|mins?|seconds?|secs?|'
    r'years?|months?|weeks?|days?|'
    r'dollars?|euros?|\$)'
)

# Verbos de actualizacion de hechos personales
_VERBOS = (
    r'\b(moved|graduated|bought|sold|got|received|started|finished|'
    r'completed|joined|left|quit|hired|fired|married|divorced|'
    r'adopted|enrolled|registered|signed|earned|won|lost|'
    r'transferred|promoted|demoted|retired|born|died|'
    r'diagnosed|prescribed|recommended)\b'
)

# Relaciones de lugar o trabajo
_RELACIONES = (
    r'\b(lives? in|moved to|works? (at|for|in)|studying at|'
    r'goes? to|attends?|enrolled (at|in)|based in)\b'
)

# Meses y dias de la semana
_FECHAS = (
    r'\b(january|february|march|april|may|june|july|august|'
    r'september|october|november|december|'
    r'monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b'
)

_PATRON = re.compile(
    '|'.join([_NUM, _TIEMPO, _UNIDADES, _VERBOS, _RELACIONES, _FECHAS]),
    re.IGNORECASE
)


def contiene_hecho_factual(texto: str) -> bool:
    """
    Retorna True si el texto contiene informacion factual concreta
    que podria contradecir una memoria guardada en Qdrant.

    Retorna False para mensajes generales como preguntas, saludos
    o conversaciones sin datos especificos.
    """
    return bool(_PATRON.search(texto))
