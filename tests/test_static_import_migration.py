from pathlib import Path

from scripts.migrate_static_imports import expected_imports, existing_imports, source_files


ROOT = Path(__file__).parents[1]


def test_every_companion_and_uses_edge_has_a_static_import():
    modules = source_files(ROOT)
    assert modules
    for path in modules:
        text = path.read_text()
        expected = expected_imports(path)
        actual = existing_imports(path, text)
        assert actual == expected, f"incomplete static import migration: {path}"
        if expected:
            # Imports must precede the legacy frontmatter delimiter and body.
            assert text.startswith("import {")
            assert "\n---\n" in text
        else:
            assert not actual


def test_import_migration_preserves_frontmatter_and_instruction_body():
    for path in source_files(ROOT):
        text = path.read_text()
        if not expected_imports(path):
            continue
        frontmatter_start = text.index("\n---\n") + 1
        frontmatter_end = text.index("\n---\n", frontmatter_start + 1) + 5
        frontmatter = text[frontmatter_start:frontmatter_end]
        body = text[frontmatter_end:]
        assert frontmatter.startswith("---\n")
        assert body
