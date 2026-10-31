from langchain.tools import ToolRuntime

from Agents import tools


def image_call(prompt: str, context=None) -> dict:
    runtime = ToolRuntime(state={}, context=context, config={}, stream_writer=None, tool_call_id="1", store=None)
    return {"type": "tool_call", "name": "generate_image", "args": {"prompt": prompt, "runtime": runtime}, "id": "1"}


def test_filesystem_tools_roundtrip(workspace):
    assert "geschrieben" in tools.write_file.invoke({"path": "notizen/a.txt", "content": "hallo"})
    assert tools.read_file.invoke({"path": "notizen/a.txt"}) == "hallo"
    assert "a.txt" in tools.list_files.invoke({"path": "notizen"})
    assert "notizen/a.txt" in tools.search_files.invoke({"pattern": "**/*.txt"})


def test_write_does_not_overwrite_without_flag(workspace):
    tools.write_file.invoke({"path": "a.txt", "content": "alt"})
    assert "existiert bereits" in tools.write_file.invoke({"path": "a.txt", "content": "neu"})
    assert tools.read_file.invoke({"path": "a.txt"}) == "alt"
    tools.write_file.invoke({"path": "a.txt", "content": "neu", "overwrite": True})
    assert tools.read_file.invoke({"path": "a.txt"}) == "neu"


def test_paths_outside_workspace_are_rejected(workspace, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("geheim")
    (workspace / "link").symlink_to(secret)

    assert "außerhalb" in tools.read_file.invoke({"path": "../secret.txt"})
    assert "außerhalb" in tools.read_file.invoke({"path": str(secret)})
    assert "außerhalb" in tools.read_file.invoke({"path": "link"})
    assert "außerhalb" in tools.write_file.invoke({"path": "../x.txt", "content": "x"})
    assert not (tmp_path / "x.txt").exists()
    assert "geheim" not in tools.search_files.invoke({"pattern": "../*"})
    assert "secret" not in tools.search_files.invoke({"pattern": "../*"})


def test_generate_image_mock_returns_artifact(workspace, monkeypatch):
    from rag_chatpipline import config

    monkeypatch.setattr(config, "IMAGE_BACKEND", "mock")
    result = tools.generate_image.invoke(image_call("a cat"))
    assert (config.IMAGES_DIR / result.artifact["name"]).read_bytes().startswith(b"\x89PNG")


def test_generate_image_reports_errors_to_the_agent(workspace, monkeypatch):
    from rag_chatpipline import config

    monkeypatch.setattr(config, "IMAGE_BACKEND", "gibtsnicht")
    result = tools.generate_image.invoke(image_call("a cat"))
    assert "fehlgeschlagen" in result.content and result.artifact == {}
