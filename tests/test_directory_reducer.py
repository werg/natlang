from natlang.runtime import Runtime, Session
from natlang.scope_surface import ScopeEvalSurface
from natlang.scoped_fs import Folder
from natlang.types import TypeEnv
from natlang.values import load_program


SOURCE = {"$lambda": {
    "type": "Lambda<{ folder: Folder }, Text>",
    "instructions": "Apply the rewrite and return its report.",
    "args": {},
    "codebase": {"rewrite": {
        "kind": "directory-reducer", "args": {"replacement": "Text"}, "returns": "Text",
        "instructions": "Replace the greeting in project/message.txt and report success.",
    }},
}}


class ReducerAgent:
    def run(self, session):
        surface = ScopeEvalSurface()
        assert session.lam.subtype == "directory-reducer"
        assert surface.apply(session, "read_file", {"path": "project/message.txt"}).value == "hello\n"
        assert surface.apply(session, "edit_file", {"path": "project/message.txt", "find": "hello",
                                                     "replace_with": session.lam.in_["replacement"]}).kind == "ok"
        assert surface.apply(session, "eval", {"code": 'const report: Text = "changed"; report'}).kind == "ok"
        assert surface.apply(session, "commit", {"value": "report"}).kind == "ok"
        assert surface.apply(session, "mark_lines", {"start": 1}).kind == "ok"
        assert surface.tools(session) == []
        assert session.finish()


def root_session(folder):
    root = load_program(SOURCE)
    root.in_["folder"] = folder
    runtime = Runtime(lambda _lam: ReducerAgent())
    return root, Session(runtime, root, TypeEnv())


def test_folder_apply_installs_directory_reducer_patch_atomically():
    folder = Folder.from_files({"message.txt": "hello\n"})
    root, session = root_session(folder)
    result = session.apply("eval", {"code": 'const report = await folder.apply(rewrite, "hi"); report'})
    assert result.kind == "done" and result.value == "changed"
    assert root.let["report"] == "changed"
    assert folder.read_text("message.txt") == "hi\n"


def test_direct_directory_reducer_call_returns_value_and_discards_patch():
    folder = Folder.from_files({"message.txt": "hello\n"})
    root, session = root_session(folder)
    result = session.apply("eval", {"code": 'const report = await rewrite(folder, "hi"); report'})
    assert result.kind == "ok" and result.value == "changed"
    assert folder.read_text("message.txt") == "hello\n"


def test_folder_and_file_handles_persist_as_typed_scope_values():
    folder = Folder.from_files({"notes/a.txt": "alpha\n"})
    root, session = root_session(folder)
    made_dir = session.apply("eval", {"code": 'const notes: Folder = folder.dir("notes"); notes'})
    assert made_dir.kind == "ok" and root.let["notes"].path == "notes"
    made_file = session.apply("eval", {"code": 'const note: FileHandle = notes.file("a.txt"); note'})
    assert made_file.kind == "ok" and root.let["note"].path == "notes/a.txt"
    read = session.apply("eval", {"code": "const text: Text = await note.readText(); text"})
    assert read.kind == "ok" and read.value == "alpha\n"
