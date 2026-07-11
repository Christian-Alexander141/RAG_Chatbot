from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_chatpipline.ragpipline import call_rag_pipeline, create_chat_session, append_exchange


def test_new_chat_session_keeps_previous_memory(tmp_path):
    memory_dir = tmp_path / "memory"

    first_session = create_chat_session(memory_dir=memory_dir)
    append_exchange(first_session["chat_id"], "Hallo", "Hallo!", memory_dir=memory_dir)

    second_session = create_chat_session(memory_dir=memory_dir)

    assert first_session["chat_id"] != second_session["chat_id"]
    assert (memory_dir / first_session["chat_id"] / "soul.md").exists()
    assert (memory_dir / second_session["chat_id"] / "soul.md").exists()

    first_memory = (memory_dir / first_session["chat_id"] / "soul.md").read_text(encoding="utf-8")
    assert "Hallo" in first_memory


def test_call_rag_pipeline_uses_existing_memory(tmp_path):
    memory_dir = tmp_path / "memory"

    session = create_chat_session(chat_id="chat-1", memory_dir=memory_dir)
    append_exchange("chat-1", "Mein Name ist Alex.", "Freut mich, Alex.", memory_dir=memory_dir)

    response = call_rag_pipeline(
        "Wie heiße ich?",
        chat_id="chat-1",
        memory_dir=memory_dir,
        llm=lambda prompt: prompt,
    )

    assert "Mein Name ist Alex" in response
    assert "chat-1" == session["chat_id"]
