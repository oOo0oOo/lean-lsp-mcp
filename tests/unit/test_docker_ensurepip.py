import runpy
import zipfile
from pathlib import Path

import pytest

refresh = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "docker/refresh_ensurepip.py")
)["refresh"]


@pytest.fixture
def seed(tmp_path):
    root = tmp_path / "ensurepip"
    bundled = root / "_bundled"
    bundled.mkdir(parents=True)
    (root / "__init__.py").write_text('_PIP_VERSION = "25.0.1"\n', encoding="utf-8")
    (bundled / "pip-25.0.1-py3-none-any.whl").write_bytes(b"old seed")
    (bundled / "other.whl").write_bytes(b"unrelated")
    return root


def make_wheel(tmp_path, name="pip", version="26.2.2"):
    wheel = tmp_path / "pip-26.2.2-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "pip-26.2.2.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n"
        )
    return wheel


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_refresh_updates_seed_and_removes_only_old_pip_wheels(tmp_path, seed):
    wheel = make_wheel(tmp_path)
    refresh(wheel, seed, "26.2.2")
    assert (seed / "__init__.py").read_text() == "_PIP_VERSION = '26.2.2'\n"
    assert (seed / "_bundled" / wheel.name).read_bytes() == wheel.read_bytes()
    assert not (seed / "_bundled/pip-25.0.1-py3-none-any.whl").exists()
    assert (seed / "_bundled/other.whl").read_bytes() == b"unrelated"
    before = snapshot(seed)
    refresh(wheel, seed, "26.2.2")
    assert snapshot(seed) == before


@pytest.mark.parametrize("name,version", [("other", "26.2.2"), ("pip", "25.0.1")])
def test_wrong_wheel_identity_leaves_seed_unchanged(tmp_path, seed, name, version):
    wheel = make_wheel(tmp_path, name, version)
    before = snapshot(seed)
    with pytest.raises(ValueError, match="identity"):
        refresh(wheel, seed, "26.2.2")
    assert snapshot(seed) == before


def test_wrong_wheel_filename_leaves_seed_unchanged(tmp_path, seed):
    wheel = make_wheel(tmp_path).rename(tmp_path / "other.whl")
    before = snapshot(seed)
    with pytest.raises(ValueError, match="filename"):
        refresh(wheel, seed, "26.2.2")
    assert snapshot(seed) == before


@pytest.mark.parametrize("source", ["# changed layout\n", "_PIP_VERSION = '1'\n" * 2])
def test_changed_python_layout_leaves_seed_unchanged(tmp_path, seed, source):
    wheel = make_wheel(tmp_path)
    (seed / "__init__.py").write_text(source, encoding="utf-8")
    before = snapshot(seed)
    with pytest.raises(ValueError, match="version declaration"):
        refresh(wheel, seed, "26.2.2")
    assert snapshot(seed) == before
