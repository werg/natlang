from pathlib import Path


PROMPTS = Path(__file__).parents[1] / "natlang" / "prompts"
FILE_TOOLS = (
    "list_files", "search_files", "read_file", "write_file", "edit_file", "diff_files",
)


def test_scope_prompts_name_the_file_tools_and_forbid_filesystem_imports():
    for name in ("tools_explicit.md", "tools_small.md", "tools_teacher_compact.md"):
        prompt = (PROMPTS / name).read_text()
        for tool in FILE_TOOLS:
            assert tool in prompt, (name, tool)
        assert "only filesystem interface" in prompt
        assert "Never import `fs`" in prompt
        assert "any other filesystem module in eval" in prompt


def test_scope_prompts_encourage_live_codebase_edits_but_preserve_its_manifest():
    for name in ("tools_explicit.md", "tools_small.md", "tools_teacher_compact.md"):
        prompt = (PROMPTS / name).read_text()
        assert "Inspect `codebase/` proactively" in prompt
        assert "Edit the instructions or crisp code" in prompt
        assert "existing" in prompt
        assert "never create, delete, rename, or move codebase files" in prompt or (
            "may not create, delete, rename, or move codebase files" in prompt
        )
