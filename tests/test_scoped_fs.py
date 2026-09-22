import pytest

from natlang.scoped_fs import Folder, FolderBusyError, FolderConflictError
from natlang.types import TypeEnv, parse_type
from natlang.values import coerce


def test_folder_handles_overlay_and_safe_paths():
    folder = Folder.from_files({"src/a.ts": "one\n", "src/b.ts": "two\n", "README.md": "guide"})
    assert [entry.path for entry in folder.list("src")] == ["src/a.ts", "src/b.ts"]
    assert [entry.path for entry in folder.dir("src").folders()] == []
    assert [entry.relative_path for entry in folder.dir().walk()] == ["README.md", "src", "src/a.ts", "src/b.ts"]
    assert folder.file("src/a.ts").read_text(1, 1) == "one\n"
    assert [match.path for match in folder.search("two")] == ["src/b.ts"]

    child = folder.fork()
    child.file("src/a.ts").edit_text("one", "ONE")
    child.file("src/b.ts").move_to(child.dir("src/archive"))
    child.file("new.txt").write_text("new")
    child.file("README.md").remove()
    diff = child.diff()
    assert [(item.path, item.kind) for item in diff.changes] == [
        ("README.md", "deleted"), ("new.txt", "added"),
        ("src/a.ts", "modified"), ("src/archive/b.ts", "added"),
        ("src/b.ts", "deleted")]
    assert diff.moves == (("src/b.ts", "src/archive/b.ts"),)
    assert folder.file("src/a.ts").read_text() == "one\n"

    installed = folder.install_from(child, include=["src/**"])
    assert [item.path for item in installed.changes] == ["src/a.ts", "src/archive/b.ts", "src/b.ts"]
    assert folder.file("src/a.ts").read_text() == "ONE\n"
    assert folder.file("src/archive/b.ts").read_text() == "two\n"
    assert folder.file("README.md").read_text() == "guide"
    assert coerce(folder, parse_type("Folder"), TypeEnv(), yaml=False, path="args/folder") is folder
    assert coerce(folder.file("README.md"), parse_type("File"), TypeEnv(), yaml=False,
                  path="args/file").relative_path == "README.md"


def test_fuzzy_edit_is_unambiguous_and_path_validation_is_strict():
    folder = Folder.from_files({"a.txt": "alpha   beta\ngamma\n"})
    folder.file("a.txt").edit_text("alpha beta", "changed", fuzzy=True)
    assert folder.file("a.txt").read_text() == "changed\ngamma\n"
    with pytest.raises(ValueError):
        folder.file("../a.txt")
    with pytest.raises(ValueError):
        folder.file("a\\b")
    with pytest.raises(ValueError):
        folder.write_text("/absolute", "bad")


def test_selected_install_is_atomic_and_detects_conflict():
    folder = Folder.from_files({"a": "a", "b": "b"})
    child = folder.fork()
    child.write_text("a", "A")
    child.write_text("b", "B")
    folder.write_text("a", "changed behind child")
    with pytest.raises(FolderConflictError):
        folder.install_from(child)
    assert folder.file("b").read_text() == "b"


def test_writer_lock_supports_nonblocking_exclusion():
    folder = Folder.from_files({"a": "a"})
    held = folder.writer().acquire(blocking=False)
    try:
        with pytest.raises(FolderBusyError):
            folder.writer().acquire(blocking=False)
    finally:
        held.release()


def test_transaction_holds_parent_lock_and_commits_without_reacquiring():
    folder = Folder.from_files({"a": "a", "b": "b"})
    transaction = folder.begin_transaction(blocking=False)
    transaction.folder.write_text("a", "A")
    with pytest.raises(FolderBusyError):
        folder.writer().acquire(blocking=False)
    installed = transaction.commit(include=["a"])
    assert [change.path for change in installed.changes] == ["a"]
    assert folder.file("a").read_text() == "A"
    assert folder.file("b").read_text() == "b"

    aborted = folder.begin_transaction(blocking=False)
    aborted.folder.write_text("b", "B")
    aborted.abort()
    assert folder.file("b").read_text() == "b"

    nested = folder.dir("nested").begin_transaction(blocking=False)
    nested.folder.write_text("inside.txt", "scoped")
    nested.commit()
    assert folder.file("nested/inside.txt").read_text() == "scoped"
