"""Registry of directly trained Neuralese artifacts (plans/neuralese/S8_TARGET.md §6, IMPLEMENTATION_BACKLOG.md).

An artifact is a `.nz` file (or a small set of files) a run produces or consumes as a value rather than as model
weights: system-prompt banks, the standard library of combinator soft bodies, operators, soft skills, data blocks,
adapters and projections. Like corpora (training/neuralese_corpora.json), artifacts are registered in
`training/neuralese_artifacts.json` and pinned by an immutable SHA-256 manifest in `training/artifact-manifests/`.
An entry also pins what an artifact means: its dialect (every block's dialect must equal it), the backbone and
revision its space belongs to, how it was initialised (text, a parent artifact, conversion), what trained it
(registered corpora, trainer, commit), and its qualification. A registered artifact is not admitted for use by
registration: qualification stays explicit.

Runs refer to artifacts by id (recipe input binding `{"artifact": ID}`; `resolve`), which yields the stored path
after checking its bytes against the manifest. Pure Python (no torch), so launchers and validation stay light.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
REGISTRY = "training/neuralese_artifacts.json"
MANIFESTS = "training/artifact-manifests"
STORE = "data/neuralese/artifacts"
SCHEMA = "natlang.neuralese-artifacts/1"
SNAPSHOT_SCHEMA = "natlang.neuralese-artifact-snapshot/1"
# "bundle": one file holding several kinds, e.g. a method-arm or memetic run's soft skills and adapters.
KINDS = {"prompt-bank", "standard-library", "operator", "soft-skill", "data-block", "adapter", "projection", "bundle"}
INIT_METHODS = {"text", "trained", "converted", "written"}
QUALIFICATION = {"unqualified", "qualified", "failed", "superseded"}
ID = re.compile(r"[a-z0-9][a-z0-9.-]{2,127}")
DIALECT = re.compile(r"nd:[a-z0-9-]+@[0-9]+(\.[0-9]+)*")


class ArtifactError(ValueError):
    pass


def empty_registry() -> dict:
    return {"schema": SCHEMA,
            "policy": ("Directly trained Neuralese artifacts by id. Bytes are pinned by immutable manifests in "
                       f"{MANIFESTS}; a changed artifact gets a new id. Dialect and backbone are part of identity; "
                       "registration is not qualification."),
            "artifacts": []}


def load_registry(repo: Path = REPO) -> dict:
    path = repo / REGISTRY
    if not path.exists():
        return empty_registry()
    registry = json.loads(path.read_text())
    if registry.get("schema") != SCHEMA:
        raise ArtifactError(f"unknown artifact registry schema: {registry.get('schema')}")
    return registry


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def entry(registry: dict, identity: str) -> dict:
    found = [a for a in registry["artifacts"] if a["id"] == identity]
    if len(found) != 1:
        raise ArtifactError(f"unknown artifact: {identity}")
    return found[0]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nz_header(path: Path) -> dict:
    """The safetensors header of a `.nz` file: its natlang metadata (exports, blocks) and tensor shapes."""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if length > 1 << 30:
            raise ArtifactError(f"not a safetensors file: {path}")
        raw = json.loads(stream.read(length))
    meta = raw.pop("__metadata__", {}) or {}
    natlang = json.loads(meta["natlang"]) if "natlang" in meta else None
    if natlang is None:
        raise ArtifactError(f"no natlang header in {path}")
    return {"natlang": natlang, "tensors": {k: v["shape"] for k, v in raw.items()}}


def nz_summary(path: Path) -> dict:
    header = nz_header(path)
    blocks = header["natlang"].get("blocks", {})
    widths = sorted({shape[-1] for shape in header["tensors"].values() if shape})
    return {"exports": sorted(header["natlang"].get("exports", {})),
            "blocks": len(blocks),
            "dialects": sorted({b.get("dialect") for b in blocks.values() if b.get("dialect")}),
            "widths": widths}


def safe_relative(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ArtifactError(f"unsafe relative path: {value}")
    return str(path)


def validate_entry(item: dict, registry: dict, corpora_ids: set[str] | None = None) -> None:
    """Structural and referential checks of one registry entry (bytes are checked by the manifest)."""
    if not ID.fullmatch(item.get("id", "")):
        raise ArtifactError(f"artifact id must match {ID.pattern}: {item.get('id')}")
    if item.get("kind") not in KINDS:
        raise ArtifactError(f"artifact kind must be one of {sorted(KINDS)}")
    if not DIALECT.fullmatch(item.get("dialect", "")):
        raise ArtifactError("artifact dialect must look like nd:natlang@1")
    backbone = item.get("backbone")
    if not isinstance(backbone, dict) or not backbone.get("model") or not backbone.get("revision"):
        raise ArtifactError("artifact backbone needs model and revision (the space the blocks live in)")
    init = item.get("init")
    if not isinstance(init, dict) or init.get("method") not in INIT_METHODS:
        raise ArtifactError(f"artifact init.method must be one of {sorted(INIT_METHODS)}")
    parent = init.get("parent")
    if parent is not None:
        entry(registry, parent)
    if init["method"] in {"trained", "converted"} and parent is None and not init.get("source"):
        raise ArtifactError("a trained or converted artifact names its parent artifact or source")
    training = item.get("training")
    if init["method"] == "trained":
        if not isinstance(training, dict) or not training.get("trainer") or not training.get("commit"):
            raise ArtifactError("a trained artifact records training.trainer and training.commit")
        unknown = set(training.get("corpora", [])) - (corpora_ids or set())
        if corpora_ids is not None and unknown:
            raise ArtifactError(f"training corpora are not registered: {sorted(unknown)}")
    extra = item.get("extra_dialects", [])
    if not isinstance(extra, list) or any(not isinstance(d, str) or not d.startswith("adapter/") for d in extra):
        raise ArtifactError("extra_dialects lists adapter dialects (adapter/…) only")
    qualification = item.get("qualification")
    if not isinstance(qualification, dict) or qualification.get("status") not in QUALIFICATION:
        raise ArtifactError(f"artifact qualification.status must be one of {sorted(QUALIFICATION)}")
    if not item.get("files") or any(safe_relative(f) != f for f in item["files"]):
        raise ArtifactError("artifact files must be relative names")
    safe_relative(item.get("path", ""))


def registered_corpora(repo: Path = REPO) -> set[str]:
    path = repo / "training/neuralese_corpora.json"
    if not path.exists():
        return set()
    return {c["id"] for c in json.loads(path.read_text()).get("corpora", [])}


def manifest_path(repo: Path, identity: str) -> Path:
    if not ID.fullmatch(identity):
        raise ArtifactError(f"bad artifact id: {identity}")
    return repo / MANIFESTS / f"{identity}.json"


def snapshot(repo: Path, item: dict) -> dict:
    root = repo / item["path"]
    rows = []
    for name in sorted(item["files"]):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"missing artifact file: {path}")
        row = {"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix == ".nz":
            row["nz"] = nz_summary(path)
            # Adapter blocks carry their own dialect (adapter/1;base=…), declared in extra_dialects.
            allowed = {item["dialect"], *item.get("extra_dialects", [])}
            wrong = [d for d in row["nz"]["dialects"] if d not in allowed]
            if wrong or not row["nz"]["dialects"]:
                raise ArtifactError(f"{name}: block dialects {row['nz']['dialects']} differ from {item['dialect']}")
        rows.append(row)
    return {"schema": SNAPSHOT_SCHEMA, "id": item["id"], "kind": item["kind"], "dialect": item["dialect"],
            **({"extra_dialects": item["extra_dialects"]} if item.get("extra_dialects") else {}),
            "backbone": item["backbone"], "path": item["path"], "files": rows,
            "bytes": sum(r["bytes"] for r in rows)}


def publish(repo: Path, item: dict) -> dict:
    """Write the immutable manifest of a registered artifact (idempotent; a change needs a new id)."""
    value = snapshot(repo, item)
    target = manifest_path(repo, item["id"])
    if target.exists() and json.loads(target.read_text()) != value:
        raise ArtifactError(f"immutable artifact changed: {item['id']}; register a new id")
    save_json(target, value)
    return value


def register(repo: Path, item: dict, sources: dict[str, Path]) -> dict:
    """Copy `sources` (file name → path) into the store, add the entry and publish its manifest."""
    registry = load_registry(repo)
    if any(a["id"] == item["id"] for a in registry["artifacts"]):
        raise ArtifactError(f"artifact already registered: {item['id']}")
    item = {**item, "path": item.get("path") or f"{STORE}/{item['id']}", "files": sorted(sources)}
    validate_entry(item, registry, registered_corpora(repo))
    root = repo / item["path"]
    root.mkdir(parents=True, exist_ok=True)
    for name, source in sources.items():
        target = root / safe_relative(name)
        if target.exists():
            if digest(target) != digest(Path(source)):
                raise ArtifactError(f"store already holds different bytes at {target}")
            continue
        shutil.copyfile(source, target)
        target.chmod(0o444)
    manifest = publish(repo, item)
    registry["artifacts"].append(item)
    registry["artifacts"].sort(key=lambda a: a["id"])
    save_json(repo / REGISTRY, registry)
    return manifest


def verify(repo: Path, identity: str) -> dict:
    """Check the stored bytes and header of an artifact against its immutable manifest."""
    item = entry(load_registry(repo), identity)
    manifest = json.loads(manifest_path(repo, identity).read_text())
    current = snapshot(repo, item)
    if current != manifest:
        raise ArtifactError(f"artifact {identity} differs from its manifest")
    return manifest


REPLICAS = Path(os.environ.get("NATLANG_ARTIFACT_REPLICAS",
                               os.path.expanduser("~/.config/natlang/artifact-replicas.json")))


def local_replica(identity: str, file: str) -> Path | None:
    """This machine's declared local copy of an artifact file: ``REPLICAS`` maps artifact ids to directories holding
    the same file names. Machine-local configuration, not registry content; ``resolve`` checks the bytes."""
    try:
        directories = json.loads(REPLICAS.read_text())
    except (OSError, ValueError):
        return None
    directory = directories.get(identity)
    return Path(directory) / safe_relative(file) if directory else None


def resolve(identity: str, file: str | None = None, repo: Path | None = None, *, dialect: str | None = None,
            backbone_model: str | None = None) -> tuple[Path, str]:
    """(path, sha256) of a registered artifact's file after checking its bytes against the manifest. ``dialect`` and
    ``backbone_model``, when given, must match the entry: a value from another space is refused, not converted."""
    repo = repo or REPO
    registry = load_registry(repo)
    item = entry(registry, identity)
    if dialect and item["dialect"] != dialect:
        raise ArtifactError(f"artifact {identity} is dialect {item['dialect']}, the run needs {dialect}")
    if backbone_model and item["backbone"]["model"] != backbone_model:
        raise ArtifactError(f"artifact {identity} belongs to {item['backbone']['model']}, not {backbone_model}")
    manifest = json.loads(manifest_path(repo, identity).read_text())
    rows = {r["path"]: r for r in manifest["files"]}
    if file is None:
        if len(rows) != 1:
            raise ArtifactError(f"artifact {identity} has several files; name one of {sorted(rows)}")
        file = next(iter(rows))
    if file not in rows:
        raise ArtifactError(f"artifact {identity} has no file {file}")
    replica = local_replica(identity, file)
    if replica is not None and replica.is_file() and digest(replica) == rows[file]["sha256"]:
        # A declared fast copy of the same bytes (e.g. on NVMe while the canonical bytes sit on the slow external
        # disk): used only when it matches the manifest.
        return replica, rows[file]["sha256"]
    path = (repo / item["path"] / file).resolve()
    actual = digest(path)
    if actual != rows[file]["sha256"]:
        raise ArtifactError(f"artifact {identity}/{file} content hash mismatch")
    return path, actual


def find_by_sha(sha256: str, repo: Path | None = None) -> tuple[str, str] | None:
    """The registered (artifact id, file) whose manifest pins these bytes, if any."""
    repo = repo or REPO
    for item in load_registry(repo)["artifacts"]:
        path = manifest_path(repo, item["id"])
        if path.exists():
            for row in json.loads(path.read_text())["files"]:
                if row["sha256"] == sha256:
                    return item["id"], row["path"]
    return None


def backbone_identity(model: str, revision: str | None = None) -> dict:
    """{"model", "revision"} of a backbone: an explicit revision, a hub snapshot directory's revision, or the revision a
    local weights directory recorded when its files were verified (weights-verified.json). Refuses to guess."""
    if revision:
        return {"model": model, "revision": revision}
    parts = Path(model).parts
    if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
        return {"model": model, "revision": parts[parts.index("snapshots") + 1]}
    verified = Path(model) / "weights-verified.json"
    if verified.exists():
        recorded = json.loads(verified.read_text()).get("revision")
        if recorded:
            return {"model": model, "revision": recorded}
    raise ArtifactError(f"cannot pin the revision of backbone {model}; pass it explicitly")


def register_output(path: Path, *, identity: str, kind: str, dialect: str, backbone: dict, trainer: str, commit: str,
                    corpora: list[str] | None = None, parent: str | None = None, run: str | None = None,
                    notes: str | None = None, repo: Path | None = None) -> dict:
    """Register a file a trainer produced (init method "trained"), unqualified until a gate says otherwise."""
    import datetime

    repo = repo or REPO
    item = {"id": identity, "kind": kind, "owner": "dgx", "dialect": dialect, "backbone": backbone,
            "init": {"method": "trained", **({"parent": parent} if parent else {"source": run or str(path)})},
            "training": {"corpora": corpora or [], "trainer": trainer, "commit": commit, **({"run": run} if run else {})},
            "qualification": {"status": "unqualified", "evidence": []}, "origin": {Path(path).name: str(path)},
            "registered": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            **({"notes": notes} if notes else {})}
    return register(repo, item, {Path(path).name: Path(path)})


def corpora_by_sha(shas: set[str], repo: Path | None = None) -> dict[str, str]:
    """sha256 → registered corpus id, for files a run read (scans training/corpus-manifests)."""
    repo = repo or REPO
    found = {}
    for manifest in sorted((repo / "training/corpus-manifests").glob("*.json")):
        try:
            value = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in value.get("files", []):
            if row.get("sha256") in shas:
                found.setdefault(row["sha256"], value.get("id", manifest.stem))
    return found


def register_run(run: Path, identity: str, *, backbone_revision: str | None = None, trainer: str | None = None,
                 commit: str, repo: Path | None = None, file: str = "system-prompts.nz") -> dict:
    """Register the bank a finished trainer run wrote (``summary.json`` + ``system-prompts.nz``): dialect from the file,
    parent found by content among registered artifacts, training corpora found by the content of the run's inputs."""
    repo = repo or REPO
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    options = summary.get("options", {})
    path = run / file
    if not path.exists():
        raise ArtifactError(f"{run} has no {file} (the run trained no bank)")
    dialects = nz_summary(path)["dialects"]
    if len(dialects) != 1:
        raise ArtifactError(f"{path} mixes dialects {dialects}")
    inputs = {options[k] for k in ("records", "pieces", "prompts", "text_data") if options.get(k)}
    shas = {digest(Path(p)) for p in inputs if Path(p).exists()}
    corpora = sorted(set(corpora_by_sha(shas, repo).values()))
    bank = options.get("bank") or options.get("soft_prompts")
    parent = find_by_sha(digest(Path(bank)), repo) if bank and Path(bank).exists() else None
    backbone = summary.get("backbone") or {}
    model = backbone.get("model") or backbone.get("base") or options.get("base")
    if not model:
        raise ArtifactError(f"{run} does not record its backbone; register it with the CLI")
    return register_output(path, identity=identity, kind="prompt-bank", dialect=dialects[0],
                           backbone=backbone_identity(model, backbone_revision or backbone.get("revision")),
                           trainer=trainer or summary.get("trainer") or "natlang_neuralese.train.trajectories",
                           commit=commit, corpora=corpora, parent=parent[0] if parent else None,
                           run=str(run.resolve()), notes=f"inputs not registered as corpora: "
                           f"{len(shas) - len(corpora_by_sha(shas, repo))} of {len(shas)}", repo=repo)
