from __future__ import annotations

import os
import sys
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Awaitable, Callable

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from tests.conftest import _server_environment


async def _collect_logs(
    repo_root,
    env_overrides: dict[str, str],
    interaction: Callable[[ClientSession], Awaitable[None]],
) -> str:
    env = _server_environment(repo_root)
    env.update(env_overrides)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as errlog:
        server = StdioServerParameters(
            command=sys.executable,
            args=["-m", "lean_lsp_mcp", "--transport", "stdio"],
            env=env,
            cwd=str(repo_root),
        )

        async with stdio_client(server, errlog=errlog) as (read_stream, write_stream):
            session = ClientSession(read_stream, write_stream)
            async with session:
                await session.initialize()
                await interaction(session)

        errlog.seek(0)
        return errlog.read()


@pytest.mark.asyncio
async def test_no_stderr_when_log_level_none(repo_root, test_project_path) -> None:
    async def interaction(session: ClientSession) -> None:
        await session.list_tools()

    logs = await _collect_logs(
        repo_root,
        {
            "LEAN_LOG_LEVEL": "NONE",
            "LEAN_PROJECT_PATH": str(test_project_path),
        },
        interaction,
    )

    assert not logs.strip()


@pytest.mark.asyncio
async def test_info_level_emits_server_logs(repo_root, test_project_path) -> None:
    async def interaction(session: ClientSession) -> None:
        await session.list_tools()

    logs = await _collect_logs(
        repo_root,
        {
            "LEAN_LOG_LEVEL": "INFO",
            "LEAN_PROJECT_PATH": str(test_project_path),
        },
        interaction,
    )

    normalized = " ".join(logs.split())
    assert "Closing Lean LSP client" in normalized


@pytest.mark.asyncio
async def test_error_level_suppresses_info_logs(repo_root, test_project_path) -> None:
    async def interaction(session: ClientSession) -> None:
        await session.list_tools()

    logs = await _collect_logs(
        repo_root,
        {
            "LEAN_LOG_LEVEL": "ERROR",
            "LEAN_PROJECT_PATH": str(test_project_path),
        },
        interaction,
    )

    normalized = " ".join(logs.split())
    assert "Closing Lean LSP client" not in normalized


def test_logging_to_file(tmp_path):
    # Prepare config and log file paths
    cfg_path = tmp_path / "logcfg.json"
    log_file = tmp_path / "test.log"

    # Minimal dictConfig that writes INFO logs to the file
    cfg = {
        "version": 1,
        "formatters": {"simple": {"format": "%(levelname)s:%(message)s"}},
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": str(log_file),
            }
        },
        "root": {"level": "INFO", "handlers": ["file"]},
    }
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    # Script executed in a subprocess: loads config from LEAN_LOG_FILE_CONFIG and logs
    script = textwrap.dedent(
        """
        import os, json, logging, logging.config
        cfg_file = os.environ["LEAN_LOG_FILE_CONFIG"]
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = json.loads(f.read())
        logging.config.dictConfig(cfg)
        logging.getLogger(__name__).info("happy_path_log")
    """
    )

    env = os.environ.copy()
    env["LEAN_LOG_FILE_CONFIG"] = str(cfg_path)

    subprocess.run([sys.executable, "-c", script], check=True, env=env)

    content = Path(log_file).read_text(encoding="utf-8")
    assert "happy_path_log" in content


def test_logging_to_yaml_file(tmp_path):
    yaml = pytest.importorskip("yaml")

    cfg_path = tmp_path / "logcfg.yaml"
    log_file = tmp_path / "test_yaml.log"

    cfg = {
        "version": 1,
        "formatters": {"simple": {"format": "%(levelname)s:%(message)s"}},
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "simple",
                "filename": str(log_file),
            }
        },
        "root": {"level": "INFO", "handlers": ["file"]},
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    script = textwrap.dedent(
        """
        import os, json, logging, logging.config
        import yaml
        cfg_file = os.environ["LEAN_LOG_FILE_CONFIG"]
        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f.read())
        logging.config.dictConfig(cfg)
        logging.getLogger(__name__).info("happy_path_yaml")
    """
    )

    env = os.environ.copy()
    env["LEAN_LOG_FILE_CONFIG"] = str(cfg_path)

    subprocess.run([sys.executable, "-c", script], check=True, env=env)

    content = Path(log_file).read_text(encoding="utf-8")
    assert "happy_path_yaml" in content


def test_logging_to_ini_file(tmp_path):
    cfg_path = tmp_path / "logcfg.ini"
    log_file = tmp_path / "test_ini.log"

    # fileConfig reads Python literal args, so ensure proper quoting
    args_literal = "({}, 'a')".format(repr(str(log_file)))

    ini = """
[loggers]
keys=root

[handlers]
keys=file

[formatters]
keys=simple

[logger_root]
level=INFO
handlers=file

[handler_file]
class=logging.FileHandler
level=INFO
formatter=simple
args={args}

[formatter_simple]
format=%(levelname)s:%(message)s
""".format(args=args_literal)

    cfg_path.write_text(ini, encoding="utf-8")

    script = textwrap.dedent(
        """
        import os, logging, logging.config
        cfg_file = os.environ["LEAN_LOG_FILE_CONFIG"]
        logging.config.fileConfig(cfg_file)
        logging.getLogger(__name__).info("happy_path_ini")
    """
    )

    env = os.environ.copy()
    env["LEAN_LOG_FILE_CONFIG"] = str(cfg_path)

    subprocess.run([sys.executable, "-c", script], check=True, env=env)

    content = Path(log_file).read_text(encoding="utf-8")
    assert "happy_path_ini" in content

