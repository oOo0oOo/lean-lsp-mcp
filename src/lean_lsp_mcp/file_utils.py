from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


_LAKEFILE_NAMES = ("lakefile.lean", "lakefile.toml")
_STDLIB_DISPLAY_ROOT = ".lean-stdlib"


@dataclass(frozen=True)
class AllowedPathRoot:
    root: Path
    display_prefix: str


@dataclass(frozen=True)
class LeanPathPolicy:
    project_root: Path
    allowed_roots: tuple[AllowedPathRoot, ...]
    stdlib_root: Path | None = None

    def _resolve_allowed_path(
        self, path: Path | str
    ) -> tuple[Path, AllowedPathRoot | None]:
        candidate = Path(path).resolve(strict=False)
        for allowed in self.allowed_roots:
            if candidate.is_relative_to(allowed.root):
                return candidate, allowed
        return candidate, None

    def contains(self, path: Path | str) -> bool:
        _, allowed = self._resolve_allowed_path(path)
        return allowed is not None

    def validate_path(self, path: Path | str) -> Path:
        candidate, allowed = self._resolve_allowed_path(path)
        if allowed is None:
            raise ValueError(
                f"Path '{candidate}' is outside the active Lean project, dependencies, and stdlib roots."
            )
        return candidate

    def display_path(self, path: Path | str) -> str:
        candidate, allowed = self._resolve_allowed_path(path)
        if allowed is None:
            raise ValueError(
                f"Path '{candidate}' is outside the active Lean project, dependencies, and stdlib roots."
            )
        relative = candidate.relative_to(allowed.root)
        if not relative.parts:
            return allowed.display_prefix or "."
        relative_text = relative.as_posix()
        if not allowed.display_prefix:
            return relative_text
        return f"{allowed.display_prefix}/{relative_text}"

    def client_relative_path(self, path: Path | str) -> str:
        candidate = self.validate_path(path)
        return os.path.relpath(candidate, self.project_root)


def valid_lean_project_path(path: Path | str) -> bool:
    path_obj = Path(path).expanduser()
    try:
        resolved = path_obj.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return False

    if not resolved.is_dir():
        return False
    if not (resolved / "lean-toolchain").is_file():
        return False
    return any((resolved / name).is_file() for name in _LAKEFILE_NAMES)


def require_lean_project_path(path: Path | str) -> Path:
    path_obj = Path(path).expanduser()
    try:
        resolved = path_obj.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"Lean project path '{path}' does not exist.") from exc
    except OSError as exc:
        raise ValueError(f"Lean project path '{path}' is invalid: {exc}") from exc

    if not valid_lean_project_path(resolved):
        raise ValueError(
            f"Lean project path '{resolved}' must contain `lean-toolchain` and either `lakefile.lean` or `lakefile.toml`."
        )
    return resolved


@lru_cache(maxsize=16)
def _stdlib_src_root(project_root: str) -> Path | None:
    try:
        completed = subprocess.run(
            ["lean", "--print-prefix"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=project_root,
            check=False,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    prefix = completed.stdout.strip()
    if not prefix:
        return None

    candidate = (Path(prefix).expanduser() / "src").resolve(strict=False)
    if candidate.exists():
        return candidate
    return None


def build_lean_path_policy(project_root: Path | str) -> LeanPathPolicy:
    resolved_root = require_lean_project_path(project_root)
    allowed_roots: list[AllowedPathRoot] = [AllowedPathRoot(resolved_root, "")]

    packages_root = resolved_root / ".lake" / "packages"
    if packages_root.is_dir():
        for package_root in sorted(packages_root.iterdir()):
            try:
                resolved_package_root = package_root.resolve(strict=True)
            except (FileNotFoundError, OSError):
                continue
            allowed_roots.append(
                AllowedPathRoot(
                    resolved_package_root,
                    f".lake/packages/{package_root.name}",
                )
            )

    stdlib_root = _stdlib_src_root(str(resolved_root))
    if stdlib_root is not None:
        allowed_roots.append(AllowedPathRoot(stdlib_root, _STDLIB_DISPLAY_ROOT))

    return LeanPathPolicy(
        project_root=resolved_root,
        allowed_roots=tuple(allowed_roots),
        stdlib_root=stdlib_root,
    )


def resolve_input_path(
    file_path: str,
    *,
    project_root: Path | None = None,
    require_exists: bool = True,
) -> Path:
    path_obj = Path(file_path).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve(strict=require_exists)
    if project_root is not None:
        return (project_root / path_obj).resolve(strict=require_exists)
    return path_obj.resolve(strict=require_exists)


def get_relative_file_path(lean_project_path: Path, file_path: str) -> Optional[str]:
    """Convert a file path into a leanclient-compatible path relative to the project root."""
    policy = build_lean_path_policy(lean_project_path)
    file_path_obj = Path(file_path)

    candidates: list[Path] = []
    if file_path_obj.is_absolute():
        if file_path_obj.exists():
            candidates.append(file_path_obj.resolve(strict=True))
    else:
        project_candidate = lean_project_path / file_path_obj
        if project_candidate.exists():
            candidates.append(project_candidate.resolve(strict=True))
        cwd_candidate = Path.cwd() / file_path_obj
        if cwd_candidate.exists():
            resolved_cwd = cwd_candidate.resolve(strict=True)
            if resolved_cwd not in candidates:
                candidates.append(resolved_cwd)

    for candidate in candidates:
        if policy.contains(candidate):
            return policy.client_relative_path(candidate)
    return None


def get_file_contents(abs_path: str | Path) -> str:
    path_obj = Path(abs_path)
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path_obj, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
        return f.read()
