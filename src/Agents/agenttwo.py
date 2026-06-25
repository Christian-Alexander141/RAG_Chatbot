from langchain.agents import create_agent
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware

# Konfiguration der FilesystemMiddleware für lokale Dateizugriffe
fs_middleware = FilesystemMiddleware(backend=StateBackend())

# Erstellen des Agenten mit Ollama
# Ersetzen Sie 'ollama:modellname' durch Ihr installiertes Modell (z.B. 'ollama:llama3')
agent = create_agent(
    model="ollama:llama3",
    tools=[...], 
    middleware=[fs_middleware],
)

# Agenten aufrufen
result = agent.invoke({"messages": [{"role": "user", "content": "Generiere ein Bild und speichere es."}]})