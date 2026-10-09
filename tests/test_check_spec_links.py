"""The markdown link checker: it finds broken files and anchors, and spec/ and skills/ stay clean."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import check_spec_links as links  # noqa: E402


def write(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_checker_flags_missing_file_and_anchor(tmp_path):
    write(tmp_path, "spec/B.md", "# Title\n\n## Some `Heading` here\n")
    a = write(
        tmp_path,
        "spec/A.md",
        "[ok](B.md) [ok2](B.md#some-heading-here) [bad](B.md#nope) [gone](C.md) [web](https://x.y/z)\n"
        "`[code](Nope.md)`\n```\n[fenced](Nope.md)\n```\n[self](#missing)\n",
    )
    broken = links.check_file(a)
    assert sorted((item.target, item.reason) for item in broken) == [
        ("#missing", "anchor not found"),
        ("B.md#nope", "anchor not found"),
        ("C.md", "file not found"),
    ]


def test_duplicate_headings_get_numbered_anchors(tmp_path):
    write(tmp_path, "spec/B.md", "## Same\n\n## Same\n")
    a = write(tmp_path, "spec/A.md", "[a](B.md#same) [b](B.md#same-1) [c](B.md#same-2)\n")
    assert [item.target for item in links.check_file(a)] == ["B.md#same-2"]


def test_spec_and_skills_links_resolve():
    strict, _other = links.check(REPO)
    assert not strict, "\n".join(item.format() for item in strict)
