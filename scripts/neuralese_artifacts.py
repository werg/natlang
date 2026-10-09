#!/usr/bin/env python3
"""Register, verify, resolve and transfer directly trained Neuralese artifacts (.nz prompt banks, standard
libraries, operators, soft skills, data blocks, adapters, projections).

Registry: training/neuralese_artifacts.json; immutable manifests: training/artifact-manifests/<id>.json; bytes:
data/neuralese/artifacts/<id>/ (gitignored, moved between machines by `push`/`pull`, never by Git). Library:
training/neuralese/natlang_neuralese/artifacts.py.

    scripts/neuralese_artifacts.py register --id ID --kind prompt-bank --dialect nd:natlang@1 \\
        --backbone-model LiquidAI/LFM2.5-350M --backbone-revision REV --init-method text \\
        --init-source ts-host/src/native/system-prompts.ts --file bank.nz=/path/to/bank.nz
    scripts/neuralese_artifacts.py verify [--id ID]
    scripts/neuralese_artifacts.py resolve --id ID [--file NAME]
    scripts/neuralese_artifacts.py list
    scripts/neuralese_artifacts.py push --host pop-os --id ID   # copy bytes, then verify on the peer
    scripts/neuralese_artifacts.py pull --host dgx --id ID      # copy bytes from the peer, then verify here
"""
import argparse
import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training/neuralese"))
from natlang_neuralese import artifacts  # noqa: E402


def parse_files(values):
    files = {}
    for value in values or []:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise SystemExit("--file must be NAME=PATH")
        files[artifacts.safe_relative(name)] = Path(path).resolve()
    if not files:
        raise SystemExit("register needs at least one --file NAME=PATH")
    return files


def git_commit(repo):
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()


def cmd_register(args):
    sources = parse_files(args.file)
    item = {
        "id": args.id, "kind": args.kind, "owner": args.owner, "dialect": args.dialect,
        "backbone": {"model": args.backbone_model, "revision": args.backbone_revision,
                     **({"family": args.backbone_family} if args.backbone_family else {})},
        "init": {"method": args.init_method,
                 **({"source": args.init_source} if args.init_source else {}),
                 **({"parent": args.parent} if args.parent else {}),
                 **({"builder": args.builder} if args.builder else {})},
        "training": ({"corpora": args.training_corpus or [], "trainer": args.trainer,
                      "commit": args.commit or git_commit(args.repo),
                      **({"recipe": args.recipe} if args.recipe else {}),
                      **({"run": args.run} if args.run else {})}
                     if args.trainer else None),
        "qualification": {"status": args.qualification, "evidence": args.evidence or []},
        "origin": {name: str(path) for name, path in sources.items()},
        "registered": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **({"notes": args.notes} if args.notes else {}),
    }
    manifest = artifacts.register(args.repo, item, sources)
    print(json.dumps({"registered": args.id, "files": len(manifest["files"]), "bytes": manifest["bytes"]}))


def cmd_verify(args):
    registry = artifacts.load_registry(args.repo)
    ids = args.ids or [a["id"] for a in registry["artifacts"]]
    failed = 0
    for identity in ids:
        try:
            manifest = artifacts.verify(args.repo, identity)
            print(json.dumps({"verified": identity, "files": len(manifest["files"])}))
        except (artifacts.ArtifactError, OSError) as error:
            failed += 1
            print(json.dumps({"failed": identity, "error": str(error)}))
    if failed:
        raise SystemExit(1)


def cmd_resolve(args):
    path, sha = artifacts.resolve(args.ids[0], args.file, args.repo, dialect=args.dialect,
                                  backbone_model=args.backbone_model)
    print(json.dumps({"id": args.ids[0], "path": str(path), "sha256": sha}))


def cmd_list(args):
    for item in artifacts.load_registry(args.repo)["artifacts"]:
        print(json.dumps({k: item.get(k) for k in ("id", "kind", "dialect", "backbone", "qualification")}))


def transfer(args, direction):
    for identity in args.ids:
        item = artifacts.entry(artifacts.load_registry(args.repo), identity)
        local = args.repo / item["path"]
        remote = f"{args.remote_repo}/{item['path']}"
        if direction == "push":
            artifacts.verify(args.repo, identity)
            subprocess.run(["ssh", args.host, "mkdir -p " + shlex.quote(remote)], check=True)
            subprocess.run(["rsync", "-a", "--ignore-existing", f"{local}/", f"{args.host}:{remote}/"], check=True)
            subprocess.run(["ssh", args.host, f"cd {shlex.quote(args.remote_repo)} && python3 scripts/neuralese_artifacts.py "
                            f"verify --id {shlex.quote(identity)}"], check=True)
        else:
            local.mkdir(parents=True, exist_ok=True)
            subprocess.run(["rsync", "-a", "--ignore-existing", f"{args.host}:{remote}/", f"{local}/"], check=True)
            artifacts.verify(args.repo, identity)
        print(json.dumps({direction: identity, "host": args.host}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["register", "verify", "resolve", "list", "push", "pull"])
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--id", action="append", dest="ids")
    parser.add_argument("--file", action="append", help="register: NAME=PATH (repeatable); resolve: NAME")
    parser.add_argument("--kind", choices=sorted(artifacts.KINDS))
    parser.add_argument("--owner", choices=["dgx", "pop"], default="dgx")
    parser.add_argument("--dialect")
    parser.add_argument("--backbone-model")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--backbone-family")
    parser.add_argument("--init-method", choices=sorted(artifacts.INIT_METHODS))
    parser.add_argument("--init-source", help="text source (code path or prompt-piece ids) or converted source")
    parser.add_argument("--parent", help="registered artifact this one starts from")
    parser.add_argument("--builder", help="script that built the artifact")
    parser.add_argument("--training-corpus", action="append", help="registered corpus id (repeatable)")
    parser.add_argument("--trainer", help="trainer module or script")
    parser.add_argument("--commit", help="trainer commit (default: HEAD)")
    parser.add_argument("--recipe")
    parser.add_argument("--run", help="run directory or receipt")
    parser.add_argument("--qualification", choices=sorted(artifacts.QUALIFICATION), default="unqualified")
    parser.add_argument("--evidence", action="append")
    parser.add_argument("--notes")
    parser.add_argument("--host", default="pop-os")
    parser.add_argument("--remote-repo", default="/home/werg/natlang")
    args = parser.parse_args(argv)
    try:
        if args.action == "register":
            if not args.ids or len(args.ids) != 1:
                raise SystemExit("register needs exactly one --id")
            args.id = args.ids[0]
            missing = [k for k in ("kind", "dialect", "backbone_model", "backbone_revision", "init_method")
                       if getattr(args, k) is None]
            if missing:
                raise SystemExit("register needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
            cmd_register(args)
        elif args.action == "verify":
            cmd_verify(args)
        elif args.action == "resolve":
            if not args.ids or len(args.ids) != 1:
                raise SystemExit("resolve needs exactly one --id")
            args.file = (args.file or [None])[0]
            cmd_resolve(args)
        elif args.action == "list":
            cmd_list(args)
        else:
            if not args.ids:
                raise SystemExit(f"{args.action} needs --id")
            transfer(args, args.action)
    except artifacts.ArtifactError as error:
        raise SystemExit(f"error: {error}")


if __name__ == "__main__":
    main()
