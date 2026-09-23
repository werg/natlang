from pathlib import Path


ROOT = Path(__file__).parents[1]
PROMPTS = ROOT / "natlang" / "prompts"
REDUCER_PROMPT = ROOT / "ts-host" / "src" / "native" / "prompt.ts"
FILE_TOOLS = (
    "list_files", "search_files", "read_file", "write_file", "edit_file", "diff_files",
)


def test_function_prompts_show_normal_typescript_calls_and_function_edits():
    for name in ("tools_explicit.md", "tools_small.md", "tools_teacher_compact.md"):
        prompt = (PROMPTS / name).read_text()
        assert "call synchronous TypeScript functions normally" in prompt
        assert "await natural-language and asynchronous TypeScript functions" in prompt
        assert "read_function" in prompt and "edit_function" in prompt


def test_directory_reducer_prompt_describes_its_file_tools():
    prompt = REDUCER_PROMPT.read_text()
    for tool in FILE_TOOLS:
        assert tool in prompt, tool
    assert 'folder.dir(path)' in prompt
    assert 'someFolder.apply(reducer, ...args)' in prompt
