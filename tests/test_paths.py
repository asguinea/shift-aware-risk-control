from __future__ import annotations

from contextual_crc.utils.paths import RUN_SUBDIRS, ensure_dir, make_run_dir


def test_ensure_dir_creates_directory(tmp_path) -> None:
    target = tmp_path / "new" / "directory"

    returned = ensure_dir(target)

    assert returned == target
    assert target.is_dir()


def test_make_run_dir_creates_expected_subdirectories(tmp_path) -> None:
    run_dir = make_run_dir(tmp_path, "test_run")

    assert run_dir == tmp_path / "test_run"
    assert run_dir.is_dir()
    for subdir in RUN_SUBDIRS:
        assert (run_dir / subdir).is_dir()
