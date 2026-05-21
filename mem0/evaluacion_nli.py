import json
import logging
import time
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0.nli").setLevel(logging.INFO)
logging.getLogger("mem0.memory.main").setLevel(logging.INFO)

from mem0 import Memory
from qdrant_client import QdrantClient
import ollama

# ============================================================
# CONFIGURACIÓN
# ============================================================
DATASET_PATH    = "E:/proyecto/LongMemEval/data/longmemeval_oracle"
RESULTADOS_PATH = "E:/proyecto/resultados_con_nli.json"
CATEGORIA       = "knowledge-update"
MAX_PREGUNTAS   = 3     # cambiar a 78 para correr toda la noche
BATCH_SIZE      = 2     # igual que la evaluación oficial de Mem0
MAX_CHARS       = 300   # truncar mensajes largos para no sobrecargar Qwen

config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "qwen2.5:7b-instruct-q4_K_M",
            "ollama_base_url": "http://localhost:11434"
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "ollama_base_url": "http://localhost:11434",
            "embedding_dims": 768
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
            "embedding_model_dims": 768
        }
    }
}

# ============================================================
# FUNCIONES
# ============================================================
def limpiar_memorias():
    c = QdrantClient("localhost", port=6333)
    for col in c.get_collections().collections:
        c.delete_collection(col.name)

def cargar_sesiones(m, sesiones, user_id):
    """
    Carga mensajes en batches de 2 como hace Mem0 oficialmente.
    Trunca mensajes largos a MAX_CHARS para no sobrecargar Qwen.
    """
    for sesion in sesiones:
        if isinstance(sesion, list):
            mensajes = sesion
        elif isinstance(sesion, dict):
            mensajes = sesion.get("messages", [])
        else:
            continue

        batch_mensajes = []
        for msg in mensajes:
            if isinstance(msg, dict) and msg.get("content"):
                role = msg.get("role", "user")
                # Truncar a MAX_CHARS para no sobrecargar Qwen
                content = msg["content"].strip()[:MAX_CHARS]
                if content and len(content) > 10:
                    batch_mensajes.append({
                        "role": role,
                        "content": content
                    })

        # Cargar en batches de 2
        for i in range(0, len(batch_mensajes), BATCH_SIZE):
            batch = batch_mensajes[i:i + BATCH_SIZE]
            if batch:
                try:
                    m.add(batch, user_id=user_id)
                except Exception as e:
                    logging.warning(f"Error al agregar batch: {e}")

def responder_pregunta(m, pregunta, user_id):
    memorias = m.search(
        query=pregunta,
        filters={"user_id": user_id},
        top_k=5
    )
    contexto = "\n".join([r["memory"] for r in memorias["results"]])

    if not contexto.strip():
        return "No relevant memories found."

    prompt = f"""Based on these user memories:
{contexto}

Answer this question concisely:
{pregunta}

Answer:"""

    respuesta = ollama.chat(
        model="qwen2.5:7b-instruct-q4_K_M",
        messages=[{"role": "user", "content": prompt}]
    )
    return respuesta["message"]["content"]

def llm_judge(pregunta, respuesta_correcta, respuesta_generada):
    """
    LLM-as-Judge estricto.
    Solo marca CORRECT si la información clave coincide exactamente.
    """
    prompt = f"""You are a strict evaluator. Compare these two answers.

Question: {pregunta}
Correct answer: {respuesta_correcta}
Generated answer: {respuesta_generada}

RULES:
- Answer CORRECT only if the generated answer contains the same key information as the correct answer
- Answer INCORRECT if the generated answer has different or wrong information
- Numbers, names, and places must match exactly
- Reply ONLY with CORRECT or INCORRECT, nothing else"""

    resultado = ollama.chat(
        model="qwen2.5:7b-instruct-q4_K_M",
        messages=[{"role": "user", "content": prompt}]
    )
    texto = resultado["message"]["content"].strip().upper()
    return 1 if texto.startswith("CORRECT") else 0

# ============================================================
# EVALUACIÓN PRINCIPAL
# ============================================================
print("Cargando dataset...")
with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

preguntas_ku = [
    d for d in data
    if d['question_type'] == CATEGORIA
][:MAX_PREGUNTAS]

print(f"Evaluando {len(preguntas_ku)} preguntas de '{CATEGORIA}'")
print(f"Resultados en: {RESULTADOS_PATH}")
print("="*60)

resultados          = []
correctas           = 0
tiempos             = []
tiempo_inicio_total = time.time()

for i, item in enumerate(preguntas_ku):
    tiempo_inicio = time.time()
    print(f"\nPregunta {i+1}/{len(preguntas_ku)}: {item['question'][:80]}")

    # Limpiar y reiniciar Mem0
    limpiar_memorias()
    m = Memory.from_config(config)

    # Cargar sesiones en batches de 2
    user_id = f"eval_{item['question_id']}"
    cargar_sesiones(m, item['haystack_sessions'], user_id)

    # Responder pregunta
    respuesta = responder_pregunta(m, item['question'], user_id)

    # Evaluar con LLM-as-Judge estricto
    score = llm_judge(item['question'], item['answer'], respuesta)
    correctas += score

    # Medir tiempo
    tiempo_pregunta = round(time.time() - tiempo_inicio, 2)
    tiempos.append(tiempo_pregunta)

    # Guardar resultado
    resultado = {
        "question_id":        item['question_id'],
        "question":           item['question'],
        "answer_correcta":    item['answer'],
        "respuesta_generada": respuesta,
        "score":              score,
        "tiempo_segundos":    tiempo_pregunta,
    }
    resultados.append(resultado)

    # Guardar parcialmente por si se interrumpe
    with open(RESULTADOS_PATH, 'w') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    accuracy        = correctas / (i + 1)
    tiempo_promedio = sum(tiempos) / len(tiempos)
    print(f"  Correcta:  {item['answer']}")
    print(f"  Generada:  {respuesta[:100]}")
    print(f"  Score: {score} | Accuracy: {accuracy:.2%} | "
          f"Tiempo: {tiempo_pregunta:.1f}s | Promedio: {tiempo_promedio:.1f}s")

# ============================================================
# RESUMEN FINAL
# ============================================================
tiempo_total    = time.time() - tiempo_inicio_total
accuracy_final  = correctas / len(preguntas_ku)
tiempo_promedio = sum(tiempos) / len(tiempos)

resumen = {
    "total_preguntas":     len(preguntas_ku),
    "correctas":           correctas,
    "accuracy":            round(accuracy_final, 4),
    "accuracy_pct":        f"{accuracy_final:.2%}",
    "tiempo_promedio_seg": round(tiempo_promedio, 2),
    "tiempo_total_min":    round(tiempo_total / 60, 2),
    "categoria":           CATEGORIA,
    "detalle":             resultados,
}

with open(RESULTADOS_PATH, 'w') as f:
    json.dump(resumen, f, indent=2, ensure_ascii=False)

print("\n" + "="*60)
print("RESUMEN FINAL")
print("="*60)
print(f"Preguntas evaluadas : {len(preguntas_ku)}")
print(f"Respuestas correctas: {correctas}")
print(f"Accuracy            : {accuracy_final:.2%}")
print(f"Tiempo promedio/preg: {tiempo_promedio:.1f} segundos")
print(f"Tiempo total        : {tiempo_total/60:.1f} minutos")
print(f"Resultados guardados: {RESULTADOS_PATH}")
print("="*60)