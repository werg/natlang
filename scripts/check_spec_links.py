#!/usr/bin/env python3
"""Check that relative markdown links in spec/, skills/ and plans/*.md resolve to a file and anchor.

Exit status 1 when a link in spec/ or skills/ is broken (the contract documents). Broken links in plans/ are
reported but do not fail unless --all is given.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
STRICT_DIRS = ("spec", "skills")
LINK = re.compile(r"(?<!\!)\[(?:[^\[\]]|\[[^\]]*\])*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
INLINE_CODE = re.compile(r"`+[^`\n]*`+")
FENCE = re.compile(r"^\s*(```+|~~~+)")


@dataclass(frozen=True)
class Broken:
    source: Path
    line: int
    target: str
    reason: str

    def format(self, root: Path = ROOT) -> str:
        return f"{self.source.relative_to(root)}:{self.line}: {self.target}: {self.reason}"


def slug(heading: str) -> str:
    """GitHub-style heading anchor."""
    text = re.sub(r"`", "", heading.strip())
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.lower()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def prose_lines(path: Path):
    """Yield (line number, text) outside fenced code blocks."""
    fence = None
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if match:
            marker = match.group(1)[0]
            fence = None if fence == marker else (fence or marker)
            continue
        if fence is None:
            yield number, line


_anchor_cache: dict[Path, set[str]] = {}


def anchors(path: Path) -> set[str]:
    if path not in _anchor_cache:
        found: set[str] = set()
        seen: dict[str, int] = {}
        for _, line in prose_lines(path):
            heading = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
            if heading:
                base = slug(heading.group(1))
                count = seen.get(base, 0)
                seen[base] = count + 1
                found.add(base if count == 0 else f"{base}-{count}")
            for explicit in re.findall(r"<a\s+(?:name|id)=\"([^\"]+)\"", line):
                found.add(explicit)
        _anchor_cache[path] = found
    return _anchor_cache[path]


def check_file(path: Path) -> list[Broken]:
    broken: list[Broken] = []
    for number, line in prose_lines(path):
        for match in LINK.finditer(INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)):
            target = match.group(1).strip("<>")
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
                continue  # http:, mailto:, etc.
            file_part, _, fragment = target.partition("#")
            file_part = unquote(file_part)
            destination = path if not file_part else (path.parent / file_part)
            if not destination.exists():
                broken.append(Broken(path, number, target, "file not found"))
                continue
            if fragment and destination.is_file() and destination.suffix.lower() == ".md":
                if re.fullmatch(r"L\d+(C\d+)?(-L\d+(C\d+)?)?", fragment):
                    continue  # GitHub line anchor
                if unquote(fragment).lower() not in {a.lower() for a in anchors(destination)}:
                    broken.append(Broken(path, number, target, "anchor not found"))
    return broken


def documents(root: Path = ROOT) -> list[Path]:
    files = sorted((root / "spec").rglob("*.md")) + sorted((root / "skills").rglob("*.md"))
    files += sorted((root / "plans").glob("*.md"))
    return files


def check(root: Path = ROOT) -> tuple[list[Broken], list[Broken]]:
    """Return (broken links in spec/ and skills/, broken links elsewhere)."""
    strict: list[Broken] = []
    other: list[Broken] = []
    for path in documents(root):
        top = path.relative_to(root).parts[0]
        (strict if top in STRICT_DIRS else other).extend(check_file(path))
    return strict, other


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="fail on broken plans/ links too")
    args = parser.parse_args(argv)
    strict, other = check()
    for item in strict:
        print(f"BROKEN {item.format()}")
    for item in other:
        print(f"{'BROKEN' if args.all else 'note  '} {item.format()}")
    print(f"{len(strict)} broken in spec/ and skills/, {len(other)} in plans/")
    return 1 if strict or (args.all and other) else 0


if __name__ == "__main__":
    sys.exit(main())
