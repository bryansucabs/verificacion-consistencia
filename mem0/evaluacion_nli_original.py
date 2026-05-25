import json
import logging
import time
import warnings
import os
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0.nli").setLevel(logging.INFO)
logging.getLogger("mem0.memory.main").setLevel(logging.INFO)
logging.getLogger("mem0.vector_stores.qdrant").setLevel(logging.ERROR)
from mem0 import Memory
from qdrant_client import QdrantClient
import ollama

# Configuracion del experimento
DATASET_PATH    = "E:/proyecto/LongMemEval/data/longmemeval_oracle"
RESULTADOS_PATH = "E:/proyecto/resultados_prueba_nli.json"
CATEGORIA       = "knowledge-update"
MAX_PREGUNTAS   = 78
# True  -> Mem0 + Verificador de Consistencia (propuesta de tesis)
# False -> Mem0 baseline sin NLI (condicion de comparacion)
USE_NLI = True

# Configuracion de Mem0: LLM=Qwen, embedder=nomic-embed-text, vector_store=Qdrant
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

# Borra todas las colecciones de Qdrant para empezar limpio cada pregunta
def limpiar_memorias():
    c = QdrantClient("localhost", port=6333)
    for col in c.get_collections().collections:
        c.delete_collection(col.name)

# Carga en Mem0 solo los mensajes con has_answer=True de cada sesion
def cargar_sesiones(m, sesiones, user_id):
    for sesion in sesiones:
        if isinstance(sesion, list):
            mensajes = sesion
        elif isinstance(sesion, dict):
            mensajes = sesion.get("messages", [])
        else:
            continue

        for msg in mensajes:
            if not isinstance(msg, dict):
                continue
            if not msg.get("has_answer", False):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "").strip()
            if not content or len(content) < 10:
                continue
            try:
                m.add([{"role": "user", "content": content}], user_id=user_id)
            except Exception as e:
                logging.warning(f"Error al agregar mensaje: {e}")

# Busca memorias relevantes en Qdrant y le pregunta a Qwen la respuesta
def responder_pregunta(m, pregunta, user_id):
    memorias = m.search(
        query=pregunta,
        filters={"user_id": user_id},
        top_k=5
    )
    contexto = "\n".join([r["memory"] for r in memorias["results"]])

    if not contexto.strip():
        return "No relevant memories found."

    prompt = (
        "Based on these user memories:\n"
        + contexto
        + "\n\nAnswer this question concisely:\n"
        + pregunta
        + "\n\nAnswer:"
    )

    respuesta = ollama.chat(
        model="qwen2.5:7b-instruct-q4_K_M",
        messages=[{"role": "user", "content": prompt}]
    )
    return respuesta["message"]["content"]

# Evalua si la respuesta generada contiene el mismo hecho que la respuesta correcta
def llm_judge(pregunta, respuesta_correcta, respuesta_generada):
    prompt = (
        "You are a factual evaluator. Decide if the generated answer "
        "contains the same key fact as the correct answer.\n\n"
        "Question: " + str(pregunta) + "\n"
        "Correct answer: " + str(respuesta_correcta) + "\n"
        "Generated answer: " + str(respuesta_generada) + "\n\n"
        "RULES:\n"
        "- CORRECT if the generated answer conveys the same core fact\n"
        "- CORRECT even if wording differs (25:50 and 25 minutes 50 seconds are the same)\n"
        "- INCORRECT if the generated answer states a different fact or says it does not know\n"
        "- Reply ONLY with CORRECT or INCORRECT, nothing else"
    )

    resultado = ollama.chat(
        model="qwen2.5:7b-instruct-q4_K_M",
        messages=[{"role": "user", "content": prompt}]
    )
    texto = resultado["message"]["content"].strip().upper()
    return 1 if texto.startswith("CORRECT") else 0

# Evaluacion principal
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

# Retoma desde donde se quedo si el JSON ya existe
resultados = []
ya_procesados = set()
if os.path.exists(RESULTADOS_PATH):
    try:
        with open(RESULTADOS_PATH, 'r') as f:
            guardado = json.load(f)
        if isinstance(guardado, list):
            resultados = guardado
        elif isinstance(guardado, dict) and "detalle" in guardado:
            resultados = guardado["detalle"]
        ya_procesados = {r["question_id"] for r in resultados}
        print(f"Retomando: {len(ya_procesados)} preguntas ya procesadas, continuando desde donde se quedo.")
    except Exception:
        resultados = []
        ya_procesados = set()

correctas           = sum(r["score"] for r in resultados)
tiempos             = [r["tiempo_segundos"] for r in resultados]
tiempo_inicio_total = time.time()

for i, item in enumerate(preguntas_ku):
    if item['question_id'] in ya_procesados:
        print(f"Pregunta {i+1}/{len(preguntas_ku)}: ya procesada, saltando.")
        continue
    tiempo_inicio = time.time()
    print(f"\nPregunta {i+1}/{len(preguntas_ku)}: {item['question'][:80]}")

    # Limpiar y reiniciar Mem0 para cada pregunta
    limpiar_memorias()
    m = Memory.from_config(config)
    # Si USE_NLI=False desactiva el verificador para correr el baseline
    if not USE_NLI:
        m._verificador_nli = None

    user_id = f"eval_{item['question_id']}"
    n_has_answer = sum(
        1 for s in item['haystack_sessions']
        for msg in (s if isinstance(s, list) else s.get("messages", []))
        if isinstance(msg, dict) and msg.get("has_answer", False)
    )
    print(f"  Mensajes has_answer: {n_has_answer} (llamadas a Qwen para cargar)")
    cargar_sesiones(m, item['haystack_sessions'], user_id)

    # Responder pregunta
    respuesta = responder_pregunta(m, item['question'], user_id)

    # Evaluar con LLM-as-Judge
    score = llm_judge(item['question'], item['answer'], respuesta)
    correctas += score

    # Medir tiempo
    tiempo_pregunta = round(time.time() - tiempo_inicio, 2)
    tiempos.append(tiempo_pregunta)

    # Guardar resultado parcial por si se interrumpe
    resultado = {
        "question_id":        item['question_id'],
        "question":           item['question'],
        "answer_correcta":    item['answer'],
        "respuesta_generada": respuesta,
        "score":              score,
        "tiempo_segundos":    tiempo_pregunta,
    }
    resultados.append(resultado)

    with open(RESULTADOS_PATH, 'w') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    accuracy        = correctas / (i + 1)
    tiempo_promedio = sum(tiempos) / len(tiempos)
    print(f"  Correcta:  {item['answer']}")
    print(f"  Generada:  {respuesta[:100]}")
    print(f"  Score: {score} | Accuracy: {accuracy:.2%} | "
          f"Tiempo: {tiempo_pregunta:.1f}s | Promedio: {tiempo_promedio:.1f}s")

# Resumen final
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
#antihguo