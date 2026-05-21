import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0.nli").setLevel(logging.INFO)
logging.getLogger("mem0.memory.main").setLevel(logging.INFO)

from mem0 import Memory
from qdrant_client import QdrantClient

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

def limpiar_memorias():
    c = QdrantClient("localhost", port=6333)
    [c.delete_collection(col.name) for col in c.get_collections().collections]

def mostrar_memorias(m, user_id):
    memorias = m.get_all(filters={"user_id": user_id})
    print("  MEMORIAS FINALES:")
    for mem in memorias["results"]:
        print(f"    → {mem['memory']}")
    print()

m = Memory.from_config(config)

# ============================================================
# CASO 1: Cambio de ciudad
# ============================================================
print("\n" + "="*50)
print("CASO 1: Cambio de ciudad")
print("="*50)
limpiar_memorias()
m = Memory.from_config(config)
m.add("Vivo en Arequipa", user_id="caso1")
print("Memoria inicial guardada.")
m.add("Me mudé a Lima, ya no vivo en Arequipa", user_id="caso1")
mostrar_memorias(m, "caso1")

# ============================================================
# CASO 2: Cambio de estado civil
# ============================================================
print("="*50)
print("CASO 2: Cambio de estado civil")
print("="*50)
limpiar_memorias()
m = Memory.from_config(config)
m.add("Soy soltero", user_id="caso2")
print("Memoria inicial guardada.")
m.add("Me casé el mes pasado, ya no soy soltero", user_id="caso2")
mostrar_memorias(m, "caso2")

# ============================================================
# CASO 3: Cambio de carrera
# ============================================================
print("="*50)
print("CASO 3: Cambio de carrera")
print("="*50)
limpiar_memorias()
m = Memory.from_config(config)
m.add("Estudio ingeniería de sistemas", user_id="caso3")
print("Memoria inicial guardada.")
m.add("Cambié de carrera, ahora estudio medicina", user_id="caso3")
mostrar_memorias(m, "caso3")

# ============================================================
# CASO 4: Información NO contradictoria (debe quedar igual)
# ============================================================
print("="*50)
print("CASO 4: Sin contradicción (debe agregar ambas)")
print("="*50)
limpiar_memorias()
m = Memory.from_config(config)
m.add("Me gusta el fútbol", user_id="caso4")
print("Memoria inicial guardada.")
m.add("También me gusta el tenis", user_id="caso4")
mostrar_memorias(m, "caso4")