from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import logging
import os
from datetime import datetime
from rag_chatpipline.ragpipline import call_rag_pipeline as run_rag_pipeline

# === Model einrichten ===
LLM = "qwen3.5:2b"  # qwen3.5:2b hat ein Thinking-Problem -> sehr lange Antwortzeiten
model = ChatOllama(
    model=LLM,
    validate_model_on_init=True,
    temperature=0.8,
    num_predict=2048,
)

# === Logging einrichten ===
logging.basicConfig(
    filename="tool_calls.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# === kleines Test-Tool ===
@tool
def printhello() -> str:
    """Gibt 'Hello, World!' aus. Nutze dieses Tool, wenn der Nutzer
    danach fragt, 'Hello World' auszugeben oder zu testen."""
    logging.info("printhello Tool aufgerufen.")
    print("Hello, World!")
    return "Hello, World! wurde ausgegeben."


@tool
def bildgenerieren(scene_description: str) -> str:
    """Erstellt eine Illustration zu einer Romanszene und speichert sie
    in einem neuen Ordner. Nutze dieses Tool, wenn der Nutzer eine
    Illustration, ein Bild oder eine visuelle Darstellung einer Szene möchte.

    Args:
        scene_description: Die Beschreibung der Szene, die illustriert werden soll.
    """
    # 1. Ordner anlegen - immer, kein LLM nötig
    output_dir = create_folder()

    # 2. Szene -> Flux-Prompt destillieren (kommt später)
    flux_prompt = scene_description  # Platzhalter

    # 3. Bild erzeugen (Platzhalter, bis Linux-Support für Ollama-Bildgen da ist)
    image_path = generate_image_mock(flux_prompt, output_dir)

    return f"Bild gespeichert unter: {image_path}"

@tool
def call_rag_pipeline(query: str, chat_id: str | None = None, memory_dir: str | None = None) -> str:
    """Ruft die erweiterte RAG-Pipeline auf und nutzt dabei das Chat-Gedächtnis."""
    logging.info(f"call_rag_pipeline Tool aufgerufen mit Query: {query}")
    return run_rag_pipeline(query=query, chat_id=chat_id, memory_dir=memory_dir)


# === Hilfsfunktionen ===
def create_folder() -> str:
    """Erstellt einen neuen Ordner für KI-Bilder und gibt den Pfad zurück."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = os.path.join("AIbilder", timestamp)
    os.makedirs(folder_path, exist_ok=True)
    print(f"Ordner erstellt: {folder_path}")
    return folder_path


# === Mock-Implementierungen ===
def generate_image_mock(prompt: str, output_dir: str) -> str:
    print(f"[MOCK] Würde Bild generieren mit Prompt: {prompt}")
    print(f"[MOCK] Würde speichern in: {output_dir}")
    return os.path.join(output_dir, "mock_image.png")


# === Tools ans Modell binden ===
model_with_tools = model.bind_tools([printhello, bildgenerieren, call_rag_pipeline])


if __name__ == "__main__":
    print("-- Teste ChatOllama mit Tool ---")
    question = input("User: ")
    messages = [
        ("system", "Du bist ein hilfreicher Assistent mit Zugriff auf Tools."),
        ("human", question),
    ]

    response = model_with_tools.invoke(messages)

    # Hat das Modell ein Tool aufrufen wollen?
    if response.tool_calls:
        for tool_call in response.tool_calls:
            logging.info(f"Tool-Call erkannt: {tool_call['name']} mit Args: {tool_call['args']}")

            if tool_call["name"] == "printhello":
                result = printhello.invoke(tool_call["args"])
                logging.info(f"Tool-Ergebnis: {result}")
                print(f"LLM (Tool-Ergebnis): {result}")

            elif tool_call["name"] == "bildgenerieren":
                result = bildgenerieren.invoke(tool_call["args"])
                logging.info(f"Tool-Ergebnis: {result}")
                print(f"LLM (Tool-Ergebnis): {result}")

            else:
                logging.info(f"Unbekanntes Tool: {tool_call['name']}")
    else:
        print(f"LLM: {response.content}")