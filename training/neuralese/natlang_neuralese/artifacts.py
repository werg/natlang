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
KINDS = {"prompt-bank", "standard-library", "operator", "soft-skill", "data-block", "adapter", "projection"}
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
            wrong = [d for d in row["nz"]["dialects"] if d != item["dialect"]]
            if wrong or not row["nz"]["dialects"]:
                raise ArtifactError(f"{name}: block dialects {row['nz']['dialects']} differ from {item['dialect']}")
        rows.append(row)
    return {"schema": SNAPSHOT_SCHEMA, "id": item["id"], "kind": item["kind"], "dialect": item["dialect"],
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
    path = (repo / item["path"] / file).resolve()
    actual = digest(path)
    if actual != rows[file]["sha256"]:
        raise ArtifactError(f"artifact {identity}/{file} content hash mismatch")
    return path, actual
