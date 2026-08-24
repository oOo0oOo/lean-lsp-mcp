"""Unit tests for the symbol index enrichment behind lean_local_search."""

import asyncio
import time
from pathlib import Path

import pytest

from lean_lsp_mcp.file_utils import AllowedPathRoot, LeanPathPolicy
from lean_lsp_mcp.models import IndexStatus
from lean_lsp_mcp.tools import search as search_tool


SOURCE_MATCHES = [{"name": "Ns.thing_long", "kind": "theorem", "file": "A.lean"}]


def _policy(project_root: Path) -> LeanPathPolicy:
    return LeanPathPolicy(
        project_root=project_root,
        allowed_roots=(AllowedPathRoot(project_root, ""),),
    )


def _symbol(tmp_path: Path) -> dict:
    declaration = tmp_path / "Basic.lean"
    declaration.touch()
    return {"name": "Ns.thing", "location": {"path": str(declaration)}}


class _FakeClient:
    """Stands in for a running AsyncLeanLSPClient.

    It sleeps for *latency* rather than answering instantly, because the claim
    under test is about time: a search must return before the server does. A
    fake that replies immediately cannot tell a search that waits from one that
    does not, and cannot fail on a timeout too short for the project it serves.
    """

    def __init__(self, symbols=None, error=None, latency=0.0, index_ready=True):
        self.symbols = symbols or []
        self.error = error
        self.latency = latency
        self.index_ready = index_ready
        self.calls = []

    async def workspace_symbol(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self.error is not None:
            raise self.error
        timeout = kwargs["timeout"]
        await asyncio.sleep(min(self.latency, timeout))
        if self.latency > timeout:
            raise TimeoutError(f"workspace/symbol timed out after {timeout}s")
        return self.symbols, self.index_ready


@pytest.fixture(autouse=True)
def _fresh_probes():
    """Probes are process wide state keyed by project; a leaked one would
    decide the next test's result."""
    search_tool._forget_index_probes()
    yield
    search_tool._forget_index_probes()


@pytest.fixture
def quick_budget(monkeypatch):
    """Shrink the query budget so a test can watch a search give up on time.

    The ratio is what matters, not the seconds: the server here takes ten times
    the budget, as it does on a Mathlib project during the opening queries.
    """
    monkeypatch.setattr(search_tool, "INDEX_QUERY_TIMEOUT", 0.05)
    return 0.05


async def _settle(project_root: Path):
    """Wait for the request still warming the server in the background."""
    probe = search_tool._index_probes[project_root]
    if probe.task is not None:
        await asyncio.shield(probe.task)


# --- what the caller is told -------------------------------------------------


async def test_no_language_server_is_reported_as_unavailable(monkeypatch, tmp_path):
    """No server means the search could not look, which is not the same as
    looking and finding nothing. Starting one costs a lake serve boot."""
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: None)

    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, _policy(tmp_path)
    )

    assert result == SOURCE_MATCHES
    assert status is IndexStatus.unavailable
    assert not search_tool._index_probes


async def test_a_server_that_answers_at_once_is_consulted_on_the_first_search(
    monkeypatch, tmp_path
):
    """On a small project the index answers immediately, and the first search
    must carry its results. Deferring them would be a regression."""
    client = _FakeClient(symbols=[_symbol(tmp_path)], latency=0.0)
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )

    # The exact match exists only in the index, and still sorts first.
    assert [match["name"] for match in result] == ["Ns.thing", "Ns.thing_long"]
    assert status is IndexStatus.consulted


async def test_partial_index_is_reported_as_warming_not_consulted(
    monkeypatch, tmp_path
):
    """The server reports whether its index finished loading. Saying
    consulted without it would promise a completeness nobody checked, which is
    the false negative this field exists to remove."""
    client = _FakeClient(symbols=[], index_ready=False)
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )

    assert result == SOURCE_MATCHES
    assert status is IndexStatus.warming


# --- the claim about time ----------------------------------------------------


async def test_the_first_search_returns_before_the_server_does(
    monkeypatch, tmp_path, quick_budget
):
    """The whole point: a tool documented as fast must not wait out the
    server's opening queries, which ran past 120s on Mathlib.

    Timed rather than inferred. Awaiting the request would leave the status at
    warming all the same, so only the clock can tell the two apart.
    """
    client = _FakeClient(symbols=[_symbol(tmp_path)], latency=quick_budget * 10)
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    started = time.perf_counter()
    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )
    elapsed = time.perf_counter() - started

    assert result == SOURCE_MATCHES
    assert status is IndexStatus.warming
    assert elapsed < client.latency / 2, f"the search waited {elapsed:.3f}s"

    await _settle(policy.project_root)


async def test_the_abandoned_request_still_warms_the_server(
    monkeypatch, tmp_path, quick_budget
):
    """Giving up on the request must not throw the work away: it is the same
    work the next search needs, and on Mathlib it costs minutes."""
    client = _FakeClient(symbols=[_symbol(tmp_path)], latency=quick_budget * 10)
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    _first, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )
    assert status is IndexStatus.warming

    await _settle(policy.project_root)
    client.latency = 0.0  # warmed, as measured from the fourth query on

    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )

    assert "Ns.thing" in [match["name"] for match in result]
    assert status is IndexStatus.consulted


async def test_a_second_search_does_not_pile_a_request_on_a_warming_server(
    monkeypatch, tmp_path, quick_budget
):
    """A different query arriving mid warm up gets warming at once: a second
    request would not answer it any sooner and would only add load."""
    client = _FakeClient(symbols=[_symbol(tmp_path)], latency=quick_budget * 10)
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    await search_tool._with_index_matches(SOURCE_MATCHES, "thing", 10, policy)

    started = time.perf_counter()
    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "other", 10, policy
    )
    elapsed = time.perf_counter() - started

    assert result == SOURCE_MATCHES
    assert status is IndexStatus.warming
    assert len(client.calls) == 1
    assert elapsed < quick_budget, f"the second search waited {elapsed:.3f}s"

    await _settle(policy.project_root)


async def test_the_probe_asks_for_what_the_search_asked_for(monkeypatch, tmp_path):
    """The background request is the search's own, not a synthetic one, so its
    answer is usable; and wait_for_index stays zero, since blocking on index
    loading inside the request would defeat running it in the background."""
    client = _FakeClient(symbols=[_symbol(tmp_path)])
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    await search_tool._with_index_matches(SOURCE_MATCHES, "thing", 7, policy)

    query, kwargs = client.calls[0]
    assert query == "thing"
    assert kwargs == {
        "max_results": 7,
        "wait_for_index": 0.0,
        "timeout": search_tool.INDEX_PROBE_TIMEOUT,
    }


def test_the_query_budget_clears_a_warmed_mathlib_query():
    """Measured over two sessions against Mathlib: every query from the fourth
    on answered within 7.54s, and 15s covered nothing that 10s did not. The
    bound lives here so a future change has to argue with the measurement.
    """
    assert 7.54 < search_tool.INDEX_QUERY_TIMEOUT <= 15.0


# --- failure, and not pretending otherwise -----------------------------------


@pytest.mark.parametrize(
    "error", [RuntimeError("index unavailable"), TimeoutError("workspace/symbol")]
)
async def test_a_failing_lookup_is_reported_as_error_and_not_retried(
    error, monkeypatch, tmp_path
):
    """A server that cannot answer must say so consistently. Retrying on every
    search would alternate between warming and error forever, leaving the
    caller unable to conclude anything."""
    client = _FakeClient(error=error)
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    first, first_status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )
    second, second_status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )

    assert first == SOURCE_MATCHES and second == SOURCE_MATCHES
    assert first_status is IndexStatus.error
    assert second_status is IndexStatus.error
    assert len(client.calls) == 1


async def test_a_lookup_that_fails_after_the_server_warmed_still_survives(
    monkeypatch, tmp_path
):
    """The path taken once the server has proven itself has its own guard, and
    without a warm server first no test reaches it at all."""
    client = _FakeClient(symbols=[_symbol(tmp_path)])
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    _result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )
    assert status is IndexStatus.consulted

    client.error = RuntimeError("server went away")
    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )

    assert result == SOURCE_MATCHES
    assert status is IndexStatus.error


async def test_a_restarted_language_server_is_probed_again(monkeypatch, tmp_path):
    """Readiness belongs to a process. Carrying it across a restart would have
    the search report an index that died with the old one."""
    policy = _policy(tmp_path)
    first = _FakeClient(symbols=[_symbol(tmp_path)])
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: first)
    await search_tool._with_index_matches(SOURCE_MATCHES, "thing", 10, policy)

    second = _FakeClient(error=RuntimeError("still starting"))
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: second)

    result, status = await search_tool._with_index_matches(
        SOURCE_MATCHES, "thing", 10, policy
    )

    assert result == SOURCE_MATCHES
    assert status is IndexStatus.error
    assert search_tool._index_probes[policy.project_root].client is second
    assert len(second.calls) == 1


async def test_forgetting_probes_cancels_a_request_still_in_flight(
    monkeypatch, tmp_path, quick_budget
):
    """Shutdown runs this. A request with fifteen minutes left would otherwise
    keep the event loop from closing."""
    client = _FakeClient(symbols=[_symbol(tmp_path)], latency=quick_budget * 20)
    policy = _policy(tmp_path)
    monkeypatch.setattr(search_tool, "running_shared_client", lambda _root: client)

    await search_tool._with_index_matches(SOURCE_MATCHES, "thing", 10, policy)
    task = search_tool._index_probes[policy.project_root].task
    assert task is not None and not task.done()

    search_tool._forget_index_probes()
    await asyncio.sleep(0)

    assert task.cancelled() or task.done()
    assert not search_tool._index_probes
