from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import types


@contextmanager
def fake_lsp_file_operation(
    client,
    rel_path: str = "Foo.lean",
    project_path: Path | None = None,
    path_policy=None,
) -> Iterator[types.SimpleNamespace]:
    if hasattr(client, "open_file"):
        client.open_file(rel_path)
    yield types.SimpleNamespace(
        client=client,
        rel_path=rel_path,
        project_path=project_path or Path("/tmp/proj"),
        path_policy=path_policy
        or types.SimpleNamespace(validate_path=lambda path: path),
    )
