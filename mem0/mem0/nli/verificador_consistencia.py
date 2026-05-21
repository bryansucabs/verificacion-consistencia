"""
Módulo de Verificación de Consistencia mediante Inferencia de Lenguaje Natural.

Implementa el componente central de la propuesta de tesis:
integrar DeBERTa-v3-base como verificador activo de contradicciones
dentro del ciclo de escritura del sistema Mem0.

Autor: Bryan Edward Suca Jaramillo
Universidad Católica San Pablo — 2026
"""

import logging
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

UMBRAL_CONTRADICCION = 0.85
MODELO_NLI = "cross-encoder/nli-deberta-v3-base"
IDX_CONTRADICCION = 0
IDX_IMPLICACION = 1
IDX_NEUTRALIDAD = 2


class VerificadorConsistencia:
    """
    Verifica la consistencia entre un hecho atómico nuevo fi
    y las memorias candidatas Ci recuperadas por Qdrant.

    Implementa los tres componentes descritos en la Sección 4.2.2:
    1. Formación de pares de evaluación
    2. Inferencia de la relación lógica entre pares
    3. Determinación de la acción por umbrales de confianza
    """

    def __init__(self, umbral: float = UMBRAL_CONTRADICCION):
        self.umbral = umbral
        self._tokenizer = None
        self._modelo = None
        logger.info(f"VerificadorConsistencia inicializado con umbral={umbral}")

    def _cargar_modelo(self):
        if self._modelo is None:
            logger.info(f"Cargando modelo NLI: {MODELO_NLI}")
            self._tokenizer = AutoTokenizer.from_pretrained(MODELO_NLI)
            self._modelo = AutoModelForSequenceClassification.from_pretrained(MODELO_NLI)
            self._modelo.eval()
            logger.info("Modelo NLI cargado correctamente en CPU")

    def _inferir_par(self, premisa: str, hipotesis: str) -> Dict:
        self._cargar_modelo()
        inputs = self._tokenizer(
            premisa,
            hipotesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.no_grad():
            logits = self._modelo(**inputs).logits
        probabilidades = torch.softmax(logits, dim=-1)[0]
        return {
            "p_contradiccion": float(probabilidades[IDX_CONTRADICCION]),
            "p_implicacion":   float(probabilidades[IDX_IMPLICACION]),
            "p_neutralidad":   float(probabilidades[IDX_NEUTRALIDAD]),
        }

    def formar_pares(
        self,
        hecho_nuevo: str,
        memorias_candidatas: List[Dict]
    ) -> List[Tuple[str, str, str]]:
        """
        Componente 1: Formación de pares de evaluación.
        Pi = {(fi, mj) | mj ∈ Ci}

        Solo forma pares con memorias que tienen alta similitud
        semántica con el hecho nuevo (score > 0.5 de Qdrant).
        Esto evita comparar temas completamente distintos y
        reduce falsos positivos en la detección de contradicciones.
        """
        pares = []
        for memoria in memorias_candidatas:
            id_memoria = memoria.get("id", "")
            texto = memoria.get("text", "").strip()
            score = memoria.get("score", 0.0)

            # Solo comparar memorias con similitud semántica > 0.5
            if texto and score > 0.7:
                pares.append((id_memoria, hecho_nuevo, texto))

        logger.debug(f"Formados {len(pares)} pares de evaluación relevantes")
        return pares

    def inferir_relaciones(
        self,
        pares: List[Tuple[str, str, str]]
    ) -> List[Dict]:
        """
        Componente 2: Inferencia de la relación lógica entre pares.

        Para cada par (fi, mj) produce la terna pij:
        pij = (P(contradicción), P(implicación), P(neutralidad))
        Solo P(contradicción) actúa como señal activa del módulo.
        """
        if not pares:
            return []

        resultados = []
        for id_memoria, fi, mj in pares:
            try:
                probs = self._inferir_par(premisa=fi, hipotesis=mj)
                resultado = {
                    "id_memoria": id_memoria,
                    "memoria_texto": mj,
                    "hecho_nuevo": fi,
                    **probs,
                }
                resultados.append(resultado)
                logger.debug(
                    f"Par evaluado — "
                    f"P(contradicción)={probs['p_contradiccion']:.3f} | "
                    f"P(implicación)={probs['p_implicacion']:.3f} | "
                    f"P(neutralidad)={probs['p_neutralidad']:.3f}"
                )
            except Exception as e:
                logger.warning(f"Error al evaluar par id={id_memoria}: {e}")

        return resultados

    def determinar_accion(
        self,
        resultados: List[Dict]
    ) -> Tuple[bool, Optional[Dict]]:
        """
        Componente 3: Determinación de la acción por umbrales de confianza.

        Caso 1 — Contradicción detectada:
            Algún Pj(contradicción) > 0.85
            → Retorna (True, resultado_max)
            → El módulo eliminará la memoria en conflicto de Qdrant

        Caso 2 — Sin contradicción suficiente:
            Ningún Pj(contradicción) > 0.85
            → Retorna (False, None)
            → Flujo estándar de Mem0 sin modificaciones
        """
        if not resultados:
            return False, None

        resultado_max = max(resultados, key=lambda r: r["p_contradiccion"])
        p_max = resultado_max["p_contradiccion"]

        if p_max > self.umbral:
            logger.info(
                f"CONTRADICCIÓN DETECTADA — "
                f"P(contradicción)={p_max:.3f} > umbral={self.umbral}\n"
                f"  Memoria en conflicto: '{resultado_max['memoria_texto'][:80]}'\n"
                f"  Hecho nuevo:          '{resultado_max['hecho_nuevo'][:80]}'"
            )
            return True, resultado_max

        logger.debug(
            f"Sin contradicción — "
            f"P(contradicción) máx={p_max:.3f} ≤ umbral={self.umbral}"
        )
        return False, None

    def verificar(
        self,
        hecho_nuevo: str,
        memorias_candidatas: List[Dict]
    ) -> Tuple[bool, Optional[str]]:
        """
        Método principal. Ejecuta los tres componentes en secuencia.

        Args:
            hecho_nuevo: El hecho atómico fi extraído del mensaje
            memorias_candidatas: [{"id": str, "text": str, "score": float}]

        Returns:
            (hay_contradiccion, id_memoria_en_conflicto)
        """
        pares = self.formar_pares(hecho_nuevo, memorias_candidatas)
        if not pares:
            return False, None

        resultados = self.inferir_relaciones(pares)
        if not resultados:
            return False, None

        hay_contradiccion, resultado_conflicto = self.determinar_accion(resultados)

        if hay_contradiccion:
            id_idx = resultado_conflicto["id_memoria"]
            return True, id_idx

        return False, None