from tests.helpers import test_project


def test_old_mathlib_cache_requires_refresh(tmp_path):
    mathlib = tmp_path / ".lake/packages/mathlib"
    mathlib.mkdir(parents=True)
    (tmp_path / ".lake/build").mkdir()
    (mathlib / "lean-toolchain").write_text("leanprover/lean4:v4.30.0\n")

    assert test_project._should_refresh(tmp_path)

    (mathlib / "lean-toolchain").write_text(test_project.LEAN_TOOLCHAIN)
    assert not test_project._should_refresh(tmp_path)


def test_existing_manifest_does_not_skip_dependency_update(tmp_path, monkeypatch):
    (tmp_path / "lake-manifest.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        test_project.subprocess,
        "run",
        lambda args, **kwargs: calls.append((args, kwargs)),
    )

    test_project._run_lake_steps(tmp_path)

    assert [args for args, _ in calls] == [
        test_project.LAKE_UPDATE,
        *test_project.LAKE_BUILD_STEPS,
    ]
    assert all(kwargs == {"cwd": tmp_path, "check": True} for _, kwargs in calls)
