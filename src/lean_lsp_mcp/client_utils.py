import os
from pathlib import Path
from threading import Lock

from mcp.server.fastmcp import Context
from mcp.server.fastmcp.utilities.logging import get_logger
from leanclient import LeanLSPClient

from lean_lsp_mcp.file_utils import get_relative_file_path
from lean_lsp_mcp.utils import OutputCapture


logger = get_logger(__name__)
CLIENT_LOCK = Lock()


class WidgetAwareLeanLSPClient(LeanLSPClient):
    """LeanLSPClient subclass that enables widget support in interactive diagnostics.

    The standard LeanLSPClient doesn't send `hasWidgets: true` in initialization options,
    which means interactive diagnostics won't include embedded widget data (like base64
    images from #png). This subclass patches the initialization to enable widgets.
    """

    def __init__(self, project_path: str, initial_build: bool = False, prevent_cache_get: bool = False):
        # Store parameters for potential re-initialization
        self._init_project_path = project_path
        self._init_initial_build = initial_build
        self._init_prevent_cache_get = prevent_cache_get

        # Don't call super().__init__ yet - we need to patch the init sequence
        # First do the minimal setup from BaseLeanLSPClient
        from pathlib import Path
        import subprocess
        import urllib.parse
        import threading
        import asyncio
        import atexit
        import orjson

        from leanclient.utils import SemanticTokenProcessor, has_mathlib_dependency

        self.project_path = Path(project_path).resolve()
        self.request_id = 0
        self.enable_history = os.getenv("ENABLE_LEANCLIENT_HISTORY", "false").lower() == "true"
        self.history = []

        if initial_build:
            self.build_project(get_cache=not prevent_cache_get)
        elif not prevent_cache_get and has_mathlib_dependency(self.project_path):
            subprocess.run(
                ["lake", "exe", "cache", "get"],
                cwd=self.project_path,
                check=False,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        # Start LSP subprocess
        self.process = subprocess.Popen(
            ["lake", "serve"],
            cwd=self.project_path,
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.stdin = self.process.stdin
        self.stdout = self.process.stdout

        # Asyncio infrastructure
        self._loop = asyncio.new_event_loop()
        self._futures = {}
        self._notification_handlers = {}

        self._loop_thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._loop_thread.start()

        self._stdout_thread_stop_event = threading.Event()
        self._stdout_thread = threading.Thread(
            target=self._read_stdout_loop,
            args=(self._stdout_thread_stop_event,),
            daemon=True,
        )
        self._stdout_thread.start()

        # Initialize WITH hasWidgets: true
        server_info = self._send_request_sync(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": self._local_to_uri(self.project_path),
                "initializationOptions": {
                    "editDelay": 1,
                    "hasWidgets": True,  # Enable widget support!
                },
            },
        )

        legend = server_info["capabilities"]["semanticTokensProvider"]["legend"]
        self.token_processor = SemanticTokenProcessor(legend["tokenTypes"])

        self._send_notification("initialized", {})

        atexit.register(self.close)

        # Initialize file manager (from LSPFileManager.__init__)
        self.max_opened_files = 4
        self.opened_files = {}
        self._opened_files_lock = threading.Lock()
        self._close_condition = threading.Condition(self._opened_files_lock)
        self._recently_closed = set()
        self._setup_global_handlers()


def startup_client(ctx: Context):
    """Initialize the Lean LSP client if not already set up.

    Args:
        ctx (Context): Context object.
    """
    with CLIENT_LOCK:
        lean_project_path = ctx.request_context.lifespan_context.lean_project_path
        if lean_project_path is None:
            raise ValueError("lean project path is not set.")

        # Check if already correct client
        client: LeanLSPClient | None = ctx.request_context.lifespan_context.client

        if client is not None:
            # Both are Path objects now, direct comparison works
            if client.project_path == lean_project_path:
                return  # Client already set up correctly - reuse it!
            # Different project path - close old client
            client.close()

        # Need to create a new client
        # In test environments, prevent repeated cache downloads
        prevent_cache = bool(os.environ.get("LEAN_LSP_TEST_MODE"))
        # Use WidgetAwareLeanLSPClient to enable widget support (for #png etc.)
        with OutputCapture() as output:
            client = WidgetAwareLeanLSPClient(
                lean_project_path, initial_build=False, prevent_cache_get=prevent_cache
            )
            logger.info(f"Connected to Lean language server at {lean_project_path} (widget-aware)")
        build_output = output.get_output()
        if build_output:
            logger.debug(f"Build output: {build_output}")
        ctx.request_context.lifespan_context.client = client


def valid_lean_project_path(path: Path | str) -> bool:
    """Check if the given path is a valid Lean project path (contains a lean-toolchain file).

    Args:
        path (Path | str): Absolute path to check.

    Returns:
        bool: True if valid Lean project path, False otherwise.
    """
    path_obj = Path(path) if isinstance(path, str) else path
    return (path_obj / "lean-toolchain").is_file()


def infer_project_path(ctx: Context, file_path: str) -> Path | None:
    """Infer and cache the Lean project path for a file WITHOUT starting the client.

    Walks up the directory tree to find a lean-toolchain file, caches the result.
    Sets ctx.request_context.lifespan_context.lean_project_path if found.

    Side effects when path changes:
    - Next LSP tool will restart the client for the new project
    - File content hashes will be cleared

    Args:
        ctx (Context): Context object
        file_path (str): Absolute or relative path to a Lean file

    Returns:
        Path | None: The resolved project path if found, None otherwise
    """
    lifespan = ctx.request_context.lifespan_context
    if not hasattr(lifespan, "project_cache"):
        lifespan.project_cache = {}

    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    def set_project_path(project_path: Path, cache_dirs: list[str]) -> Path | None:
        """Validate file is in project, set path, update cache."""
        if get_relative_file_path(project_path, file_path) is None:
            return None

        project_path = project_path.resolve()
        lifespan.lean_project_path = project_path

        # Update all relevant directories in cache
        for directory in set(cache_dirs + [str(project_path)]):
            if directory:
                lifespan.project_cache[directory] = project_path

        return project_path

    # Fast path: current project already valid for this file
    if lifespan.lean_project_path and set_project_path(
        lifespan.lean_project_path, [file_dir]
    ):
        return lifespan.lean_project_path

    # Walk up directory tree using cache and lean-toolchain detection
    current_dir = file_dir
    while current_dir and current_dir != os.path.dirname(current_dir):
        cached_root = lifespan.project_cache.get(current_dir)

        if cached_root:
            if result := set_project_path(Path(cached_root), [current_dir]):
                return result
        elif valid_lean_project_path(current_dir):
            if result := set_project_path(Path(current_dir), [current_dir]):
                return result
        else:
            lifespan.project_cache[current_dir] = ""  # Mark as checked

        current_dir = os.path.dirname(current_dir)

    return None


def setup_client_for_file(ctx: Context, file_path: str) -> str | None:
    """Ensure the LSP client matches the file's Lean project and return its relative path."""
    project_path = infer_project_path(ctx, file_path)
    if project_path is None:
        return None

    startup_client(ctx)
    return get_relative_file_path(project_path, file_path)
