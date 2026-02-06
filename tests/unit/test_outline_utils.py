from __future__ import annotations

from lean_lsp_mcp.outline_utils import generate_outline, generate_outline_data


class DummyClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def open_file(
        self,
        path: str,
        dependency_build_mode: str = "never",
        force_reopen: bool = False,
    ) -> None:
        _ = dependency_build_mode, force_reopen
        self.calls.append(("open_file", path))

    def get_file_content(self, path: str) -> str:
        self.calls.append(("get_file_content", path))
        return ""

    def get_document_symbols(self, path: str) -> list[dict]:
        self.calls.append(("get_document_symbols", path))
        return []


def test_generate_outline_data_opens_file_by_default() -> None:
    client = DummyClient()
    generate_outline_data(client, "Foo.lean")
    assert client.calls[0] == ("open_file", "Foo.lean")


def test_generate_outline_data_can_skip_open() -> None:
    client = DummyClient()
    generate_outline_data(client, "Foo.lean", open_file=False)
    assert all(call[0] != "open_file" for call in client.calls)


def test_generate_outline_opens_file_by_default() -> None:
    client = DummyClient()
    generate_outline(client, "Foo.lean")
    assert client.calls[0] == ("open_file", "Foo.lean")


def test_generate_outline_can_skip_open() -> None:
    client = DummyClient()
    generate_outline(client, "Foo.lean", open_file=False)
    assert all(call[0] != "open_file" for call in client.calls)
