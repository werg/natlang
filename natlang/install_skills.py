"""Install the skills shipped with natlang into an explicit agent skill directory."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

NAMES = ("natlang-authoring", "natlang-integration")


def bundled_skills() -> Path:
    packaged = Path(__file__).with_name("agent_skills")
    return packaged if packaged.is_dir() else Path(__file__).resolve().parents[1] / "skills"


def install(destination: Path, names=NAMES) -> list[Path]:
    """Copy complete skills; never replace an existing installation implicitly."""
    names = tuple(dict.fromkeys(names))
    unknown = set(names) - set(NAMES)
    if unknown:
        raise ValueError(f"Unknown skills: {', '.join(sorted(unknown))}")
    source = bundled_skills()
    destination = destination.expanduser().resolve()
    targets = [destination / name for name in names]
    for name, target in zip(names, targets):
        if not (source / name / "SKILL.md").is_file():
            raise FileNotFoundError(f"Bundled skill unavailable: {name}")
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Already installed: {target}. Compare or move that directory before updating.")
    destination.mkdir(parents=True, exist_ok=True)
    for name, target in zip(names, targets):
        shutil.copytree(source / name, target)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Skill names; omit to install both")
    parser.add_argument("--dest", type=Path, help="Agent's skill discovery directory")
    parser.add_argument("--list", action="store_true", help="List bundled skills without installing")
    args = parser.parse_args()
    if args.list:
        print("\n".join(NAMES))
        return
    if args.dest is None:
        parser.error("--dest is required; select your agent's skill directory")
    try:
        for path in install(args.dest, args.names or NAMES):
            print(path)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
