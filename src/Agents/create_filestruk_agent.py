from langchain.tools import tool
from langchain.agents import create_agent
import os
from datetime import datetime


@tool
def create_folder() -> str:
    """Erstellt einen neuen Ordner für KI-Bilder und gibt den Pfad zurück."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Erstelle neuen Ordner mit Zeitstempel: {timestamp}")
    folder_path = os.path.join("AIbilder", timestamp)

    os.makedirs(folder_path, exist_ok=True)
    print(f"Ordner erstellt: {folder_path}")

    return folder_path


agent = create_agent(
    model="ollama:qwen3.5:2b",  
    tools=[create_folder],
    system_prompt=(
        "Du bist ein Assistent zum Verwalten von KI-Bildern. "
        "Wenn ein neuer Speicherordner benötigt wird, benutze das Tool create_folder."
    ),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Generiere ein Bild und speichere es."}]}
)