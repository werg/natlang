#!/usr/bin/env python3
"""Add the canonical static import block to repository-authored .nl modules.

The legacy loader derives lexical bindings from a module's companion directory and
``uses`` frontmatter.  This tool makes that graph explicit while intentionally
leaving frontmatter, instruction text, and crisp executable bodies unchanged.
"""

from __future__ import annotations

import argparse
import posixpath
import re
import sys
from pathlib import Path


ROOTS = ("codebases", "packages", "ts-host/studio/programs", "examples")
IMPORT_RE = re.compile(
    r"^import\s*\{\s*([A-Za-z_$][\w$]*)(?:\s+as\s+([A-Za-z_$][\w$]*))?\s*\}"
    r"\s*from\s*([\"'])([^\"']+)\3;?\s*$"
)
USES_RE = re.compile(r"^uses:\s*\n((?:^[ \t]+[^\n]+\n?)*)", re.MULTILINE)
USE_ITEM_RE = re.compile(r"^[ \t]+([A-Za-z_$][\w$]*):\s*(\S+)\s*$")


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in ROOTS:
        directory = root / name
        if directory.exists():
            files.extend(directory.rglob("*.nl"))
    return sorted(files)


def module_spec(path: Path, target: Path) -> tuple[str, str]:
    """Return (binding, extensionless relative module specifier)."""
    relative = posixpath.relpath(target.with_suffix(""), path.parent)
    if not relative.startswith("."):
        relative = "./" + relative
    return target.stem, relative


def expected_imports(path: Path) -> list[tuple[str, str]]:
    imports: list[tuple[str, str]] = []
    companion = path.parent / path.stem
    if companion.is_dir():
        for child in sorted(companion.iterdir()):
            if child.is_file() and child.suffix in {".nl", ".ts"}:
                # types.ts contributes aliases to the lexical type environment,
                # not a callable binding.
                if child.name != "types.ts":
                    imports.append(module_spec(path, child))

    text = path.read_text()
    match = USES_RE.search(text)
    if match:
        for item in match.group(1).splitlines():
            parsed = USE_ITEM_RE.match(item)
            if not parsed:
                raise ValueError(f"cannot parse uses entry in {path}: {item!r}")
            binding, raw_target = parsed.groups()
            specifier = posixpath.normpath(posixpath.join(".", raw_target))
            if specifier.endswith((".nl", ".ts")):
                specifier = posixpath.splitext(specifier)[0]
            if not specifier.startswith("."):
                specifier = "./" + specifier
            imports.append((binding, specifier))

    # Preserve source order for deterministic output while rejecting ambiguous
    # duplicate bindings or duplicate module edges.
    deduped: list[tuple[str, str]] = []
    seen_bindings: dict[str, str] = {}
    seen_edges: set[tuple[str, str]] = set()
    for binding, specifier in imports:
        edge = (binding, specifier)
        if edge in seen_edges:
            continue
        prior = seen_bindings.get(binding)
        if prior is not None and prior != specifier:
            raise ValueError(
                f"binding {binding!r} has conflicting imports in {path}: {prior!r}, {specifier!r}"
            )
        seen_bindings[binding] = specifier
        seen_edges.add(edge)
        deduped.append(edge)
    return deduped


def existing_imports(path: Path, text: str) -> list[tuple[str, str]]:
    imports: list[tuple[str, str]] = []
    for line in text.splitlines():
        match = IMPORT_RE.match(line)
        if match:
            imports.append((match.group(2) or match.group(1), match.group(4)))
        elif line and not line.startswith("import ") and imports:
            break
    return imports


def migrated_text(path: Path) -> str:
    text = path.read_text()
    expected = expected_imports(path)
    if not expected:
        return text
    current = existing_imports(path, text)
    def declaration(binding: str, specifier: str) -> str:
        exported = posixpath.basename(specifier)
        selected = exported if exported == binding else f"{exported} as {binding}"
        return f'import {{ {selected} }} from "{specifier}";\n'
    block = "".join(declaration(binding, specifier) for binding, specifier in expected)
    if current == expected:
        old_lines = text.splitlines(keepends=True)
        old_block = "".join(old_lines[:len(current)])
        return text if old_block == block else block + "".join(old_lines[len(current):])
    if current:
        raise ValueError(f"non-canonical import block in {path}: {current!r}")
    return block + text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write", action="store_true", help="write missing import blocks")
    parser.add_argument("--check", action="store_true", help="check that all import blocks are current")
    args = parser.parse_args()
    if args.check and args.write:
        parser.error("--check and --write are mutually exclusive")
    changed = 0
    errors: list[str] = []
    for path in source_files(args.root):
        try:
            before = path.read_text()
            after = migrated_text(path)
            if before != after:
                changed += 1
                if args.write:
                    path.write_text(after)
        except (OSError, ValueError) as error:
            errors.append(str(error))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    mode = "updated" if args.write else "would update"
    print(f"{mode} {changed} module(s); checked {len(source_files(args.root))} .nl module(s)")
    return 0 if args.write or changed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
