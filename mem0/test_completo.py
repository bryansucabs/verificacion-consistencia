import logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger("mem0.nli").setLevel(logging.INFO)
logging.getLogger("mem0.memory.main").setLevel(logging.INFO)

from mem0 import Memory

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

m = Memory.from_config(config)

print("\n--- AGREGANDO MEMORIA 1 ---")
r1 = m.add("Me llamo Bryan y trabajo en Microsoft", user_id="bryan")
print(r1)

print("\n--- AGREGANDO MEMORIA 2 (contradicción) ---")
r2 = m.add("Ahora trabajo en Google, dejé Microsoft", user_id="bryan")
print(r2)

print("\n--- MEMORIAS FINALES ---")
memorias = m.get_all(filters={"user_id": "bryan"})
for m_ in memorias["results"]:
    print(f"  → {m_['memory']}")