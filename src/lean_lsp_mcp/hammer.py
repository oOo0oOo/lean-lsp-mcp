"""Local lean-premise-server manager using Docker or macOS container."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path

import orjson

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    if d := os.environ.get("LEAN_HAMMER_CACHE_DIR"):
        return Path(d)
    xdg = os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
    return Path(xdg) / "lean-lsp-mcp" / "hammer"


class HammerManager:
    """Manages local lean-premise-server container.

    Uses Docker or macOS container tool to run the premise selection server.
    """

    IMAGE = "ghcr.io/hanwenzhu/lean-premise-server:latest"
    CONTAINER_NAME = "lean-premise-server"
    DEFAULT_PORT = 8765

    def __init__(self, port: int | None = None):
        self.port = port or int(os.environ.get("LEAN_HAMMER_PORT", self.DEFAULT_PORT))
        self.image = os.environ.get("LEAN_HAMMER_IMAGE", self.IMAGE)
        self.container_name = os.environ.get(
            "LEAN_HAMMER_CONTAINER_NAME", self.CONTAINER_NAME
        )
        self.cache_dir = get_cache_dir()
        self._container_tool: str | None = None
        self._running = False

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def _find_container_tool(self) -> str | None:
        """Find available container runtime (Docker or macOS container)."""
        if self._container_tool:
            return self._container_tool

        # Prefer macOS container tool if available
        if shutil.which("container"):
            self._container_tool = "container"
            return self._container_tool

        # Fall back to Docker
        if shutil.which("docker"):
            self._container_tool = "docker"
            return self._container_tool

        return None

    def _run_cmd(
        self, cmd: list[str], timeout: int = 60, check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a container command."""
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=check
        )

    def is_available(self) -> bool:
        """Check if container tool is available."""
        return self._find_container_tool() is not None

    def is_running(self) -> bool:
        """Check if the premise server container is running."""
        tool = self._find_container_tool()
        if not tool:
            return False

        try:
            result = self._run_cmd(
                [tool, "inspect", self.container_name], timeout=10, check=False
            )
            if result.returncode != 0:
                return False

            # Check if container is actually running
            if tool == "docker":
                result = self._run_cmd(
                    [
                        "docker",
                        "inspect",
                        "-f",
                        "{{.State.Running}}",
                        self.container_name,
                    ],
                    timeout=10,
                    check=False,
                )
                return result.stdout.strip() == "true"
            else:
                # macOS container tool
                result = self._run_cmd(
                    [tool, "inspect", self.container_name], timeout=10, check=False
                )
                return "running" in result.stdout.lower()
        except Exception:
            return False

    def _health_check(self) -> bool:
        """Check if the server is responding."""
        try:
            req = urllib.request.Request(
                f"{self.url}/health",
                headers={"User-Agent": "lean-lsp-mcp/0.1"},
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False

    def pull_image(self) -> bool:
        """Pull the premise server image."""
        tool = self._find_container_tool()
        if not tool:
            logger.error("No container runtime found")
            return False

        logger.info(f"Pulling {self.image}...")
        try:
            if tool == "docker":
                self._run_cmd(["docker", "pull", self.image], timeout=600)
            else:
                self._run_cmd([tool, "image", "pull", self.image], timeout=600)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pull image: {e.stderr}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Image pull timed out")
            return False

    def start(self) -> bool:
        """Start the premise server container."""
        tool = self._find_container_tool()
        if not tool:
            logger.error("No container runtime found (need Docker or macOS container)")
            return False

        # Check if already running
        if self.is_running():
            if self._health_check():
                logger.info("Premise server already running")
                self._running = True
                return True
            else:
                # Container running but unhealthy, restart it
                self.stop()

        # Pull image if needed
        if not self.pull_image():
            return False

        # Create cache directory for volumes
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting premise server on port {self.port}...")
        try:
            if tool == "docker":
                # Remove existing container if any
                self._run_cmd(
                    ["docker", "rm", "-f", self.container_name], timeout=30, check=False
                )
                # Start new container
                self._run_cmd(
                    [
                        "docker",
                        "run",
                        "-d",
                        "--name",
                        self.container_name,
                        "-p",
                        f"{self.port}:80",
                        "-e",
                        "MAX_BATCH_TOKENS=16384",  # CPU-friendly batch size
                        self.image,
                    ],
                    timeout=60,
                )
            else:
                # macOS container tool
                self._run_cmd(
                    [tool, "rm", "-f", self.container_name], timeout=30, check=False
                )
                self._run_cmd(
                    [
                        tool,
                        "run",
                        "-d",
                        "--name",
                        self.container_name,
                        "-p",
                        f"{self.port}:80",
                        "-e",
                        "MAX_BATCH_TOKENS=16384",
                        self.image,
                    ],
                    timeout=60,
                )

            # Wait for server to be ready
            for _ in range(30):  # Wait up to 30 seconds
                if self._health_check():
                    logger.info("Premise server started successfully")
                    self._running = True
                    return True
                asyncio.get_event_loop().run_until_complete(asyncio.sleep(1))

            logger.error("Premise server failed to become healthy")
            return False

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start container: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return False

    async def start_async(self) -> bool:
        """Async version of start()."""
        tool = self._find_container_tool()
        if not tool:
            logger.error("No container runtime found (need Docker or macOS container)")
            return False

        if self.is_running():
            if self._health_check():
                logger.info("Premise server already running")
                self._running = True
                return True
            else:
                await self.stop_async()

        # Run blocking operations in thread pool
        loop = asyncio.get_event_loop()
        if not await loop.run_in_executor(None, self.pull_image):
            return False

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting premise server on port {self.port}...")
        try:

            def _start_container():
                if tool == "docker":
                    self._run_cmd(
                        ["docker", "rm", "-f", self.container_name],
                        timeout=30,
                        check=False,
                    )
                    self._run_cmd(
                        [
                            "docker",
                            "run",
                            "-d",
                            "--name",
                            self.container_name,
                            "-p",
                            f"{self.port}:80",
                            "-e",
                            "MAX_BATCH_TOKENS=16384",
                            self.image,
                        ],
                        timeout=60,
                    )
                else:
                    self._run_cmd(
                        [tool, "rm", "-f", self.container_name],
                        timeout=30,
                        check=False,
                    )
                    self._run_cmd(
                        [
                            tool,
                            "run",
                            "-d",
                            "--name",
                            self.container_name,
                            "-p",
                            f"{self.port}:80",
                            "-e",
                            "MAX_BATCH_TOKENS=16384",
                            self.image,
                        ],
                        timeout=60,
                    )

            await loop.run_in_executor(None, _start_container)

            # Wait for server to be ready
            for _ in range(30):
                if self._health_check():
                    logger.info("Premise server started successfully")
                    self._running = True
                    return True
                await asyncio.sleep(1)

            logger.error("Premise server failed to become healthy")
            return False

        except Exception as e:
            logger.error(f"Failed to start container: {e}")
            return False

    def stop(self) -> None:
        """Stop the premise server container."""
        tool = self._find_container_tool()
        if not tool:
            return

        try:
            if tool == "docker":
                self._run_cmd(
                    ["docker", "stop", self.container_name], timeout=30, check=False
                )
            else:
                self._run_cmd(
                    [tool, "stop", self.container_name], timeout=30, check=False
                )
            self._running = False
            logger.info("Premise server stopped")
        except Exception:
            pass

    async def stop_async(self) -> None:
        """Async version of stop()."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.stop)

    def query(self, goal_state: str, num_results: int = 32) -> list[dict]:
        """Query the local premise server."""
        if not self._running and not self.is_running():
            raise RuntimeError("Premise server not running")

        data = {
            "state": goal_state,
            "new_premises": [],
            "k": num_results,
        }

        req = urllib.request.Request(
            f"{self.url}/retrieve",
            headers={
                "User-Agent": "lean-lsp-mcp/0.1",
                "Content-Type": "application/json",
            },
            method="POST",
            data=orjson.dumps(data),
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            results = orjson.loads(response.read())

        return results

    async def query_async(self, goal_state: str, num_results: int = 32) -> list[dict]:
        """Async version of query()."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, goal_state, num_results)
