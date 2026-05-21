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
m.add("Me llamo Bryan y trabajo en la UCSP", user_id="test")
print("✓ Memoria guardada")

resultados = m.get_all(filters={"user_id": "test"})
print("✓ Memorias recuperadas:", resultados)