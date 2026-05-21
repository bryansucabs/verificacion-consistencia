# Verificación de Consistencia en Mem0 mediante NLI

**Autor:** Bryan Edward Suca Jaramillo  
**Universidad:** Universidad Católica San Pablo (UCSP) — Arequipa, Perú  
**Año:** 2026  
**Asesora:** Dra. Graciela Lecireth Meza Lovón

---

## Descripción

Este repositorio contiene la implementación del módulo de verificación de consistencia propuesto en la tesis:

> *"Verificación de Consistencia en el Sistema de Memoria Mem0 para la Detección de Contradicciones mediante Inferencia de Lenguaje Natural"*

El módulo detecta y resuelve contradicciones en el sistema de memoria **Mem0** usando el modelo de lenguaje **DeBERTa-v3-base** como clasificador NLI (Natural Language Inference).

---

## Problema que resuelve

Cuando un usuario actualiza información previamente guardada, Mem0 almacena ambas versiones de forma contradictoria:

```
Sesión 1: "Trabajo en Microsoft"  → Mem0 guarda: "User works at Microsoft"
Sesión 2: "Ahora trabajo en Google" → Mem0 guarda: "User works at Google"

Resultado sin módulo:
  → User works at Microsoft    ← CONTRADICCIÓN
  → User works at Google

Resultado con módulo NLI:
  → User works at Google       ← CORRECTO
```

---

## Tecnologías utilizadas

| Componente | Tecnología |
|-----------|-----------|
| Sistema de memoria | Mem0 2.0 |
| LLM local | Qwen2.5 7B (via Ollama) |
| Embeddings | nomic-embed-text (via Ollama) |
| Vector store | Qdrant 1.18 (via Docker) |
| Modelo NLI | DeBERTa-v3-base (HuggingFace) |
| Benchmark | LongMemEval (oracle) |
| Hardware | GTX 3070 8GB + Ryzen 5 2600X + 32GB RAM |

---

## Arquitectura del módulo NLI

El módulo implementa tres componentes descritos en la Sección 4.2.2 de la tesis:

```
MENSAJE NUEVO
     ↓
[1] FORMACIÓN DE PARES
    fi = hecho nuevo extraído por Qwen
    Ci = memorias candidatas de Qdrant (score > 0.7)
    Pi = {(fi, mj) | mj ∈ Ci}
     ↓
[2] INFERENCIA DE RELACIÓN LÓGICA
    DeBERTa evalúa cada par (fi, mj):
    pij = (P(contradicción), P(implicación), P(neutralidad))
     ↓
[3] DETERMINACIÓN DE ACCIÓN
    Si P(contradicción) > 0.85:
      → Eliminar mj de Qdrant
      → Guardar fi como nueva memoria
    Si no:
      → Flujo estándar de Mem0
```

---

## Estructura del repositorio

```
├── mem0/mem0/
│   ├── memory/
│   │   └── main.py              ← Mem0 modificado con módulo NLI
│   └── nli/
│       ├── __init__.py
│       └── verificador_consistencia.py  ← Módulo NLI (tesis)
├── evaluacion_nli.py            ← Script de evaluación con LongMemEval
├── test_completo.py             ← Prueba básica del módulo
├── test_casos.py                ← Prueba con 4 casos de contradicción
└── LongMemEval/
    └── data/
        └── longmemeval_oracle   ← Dataset (500 preguntas, 78 knowledge-update)
```

---

## Requisitos

```bash
# Servicios necesarios
docker run -p 6333:6333 qdrant/qdrant
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull nomic-embed-text

# Dependencias Python
pip install mem0ai transformers torch qdrant-client ollama
```

---

## Prueba rápida del módulo

```bash
python test_completo.py
```

Salida esperada:
```
--- AGREGANDO MEMORIA 1 ---
{'results': [{'memory': 'User works at Microsoft', 'event': 'ADD'}]}

--- AGREGANDO MEMORIA 2 (contradicción) ---
CONTRADICCIÓN DETECTADA — P(contradicción)=0.991 > umbral=0.85
[NLI] Eliminando memoria en conflicto

--- MEMORIAS FINALES ---
  → User works at Google    ← solo una memoria, sin contradicción
```

---

## Evaluación con LongMemEval

La evaluación compara Mem0 original vs Mem0 + módulo NLI en la categoría **knowledge-update** (78 preguntas).

```bash
# Evaluación CON módulo NLI
python evaluacion_nli.py

# Para evaluación SIN módulo NLI:
# En mem0/mem0/memory/main.py cambiar:
# self._verificador_nli = None
# Y cambiar RESULTADOS_PATH = "resultados_sin_nli.json"
```

Métricas registradas:
- **Accuracy** (LLM-as-Judge con Qwen)
- **Tiempo promedio** por pregunta
- **Contradicciones detectadas** por el módulo

---

## Dataset LongMemEval

El dataset completo debe descargarse por separado:

```bash
# Solo el oracle (15MB) — suficiente para evaluación
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='xiaowu0162/longmemeval-cleaned',
    filename='longmemeval_oracle.json',
    repo_type='dataset',
    local_dir='LongMemEval/data'
)
"
```

---

## Referencia

Wu, D. et al. (2025). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory*. ICLR 2025.
