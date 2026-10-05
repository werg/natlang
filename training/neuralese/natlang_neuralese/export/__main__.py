"""`python -m natlang_neuralese.export --out DIR [--heads checkpoint.pt] [--cutoff 6] [--max-block 64]`

Writes `DIR/model.gguf` (the backbone with the control rows merged) and `DIR/neuralese.gguf` (the port heads) for the
llama.cpp fork (training/neuralese/llama-cpp-fork.json): `llama-neuralese-server -m DIR/model.gguf --nz DIR/neuralese.gguf`.
Without `--heads` the heads are untrained (as in the reference server), which is enough to test plumbing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


_HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path: Path) -> str:
    """Hash a file with bounded memory; callers only pass regular files."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(path: Path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file_identity(path: Path) -> dict:
    path = path.resolve()
    before = _stat_identity(path)
    digest = _sha256_file(path)
    if _stat_identity(path) != before:
        raise ValueError(f"Artifact changed while hashing: {path}")
    return {"path": str(path), "size_bytes": before[2], "sha256": digest}


def _local_artifact_identity(source: str | None) -> dict | None:
    """Describe a local LoRA artifact without fetching remote identifiers."""
    if source is None:
        return None
    root = Path(source).expanduser()
    if not root.exists():
        return {"source": source, "local_artifact": False, "sha256": None,
                "note": "Not a local path; no artifact was downloaded for provenance."}
    if root.is_file():
        return {"source": source, "local_artifact": True, **_file_identity(root)}
    if not root.is_dir():
        return {"source": source, "local_artifact": False, "sha256": None,
                "note": "Path is neither a regular file nor a directory."}

    files = []
    for current, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(current) / d).is_symlink())
        for name in sorted(names):
            path = Path(current) / name
            if path.is_file() and not path.is_symlink():
                files.append(path)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    aggregate = hashlib.sha256()
    entries = []
    for path in files:
        identity = _file_identity(path)
        relative = path.relative_to(root).as_posix()
        entries.append({"path": relative, "size_bytes": identity["size_bytes"], "sha256": identity["sha256"]})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(identity["size_bytes"]).encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(identity["sha256"].encode("ascii"))
        aggregate.update(b"\n")
    return {"source": source, "local_artifact": True, "kind": "directory",
            "file_count": len(entries), "sha256": aggregate.hexdigest(), "files": entries}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export a Neuralese port to llama.cpp GGUF files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--base", default=None)
    parser.add_argument("--lora", default=None)
    parser.add_argument("--heads", default=None, help="S3 trainer checkpoint with port heads and control rows")
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--max-block", type=int, default=None)
    parser.add_argument("--dialect", default=None)
    parser.add_argument("--outtype", default="f32", choices=["f32", "f16", "bf16"])
    args = parser.parse_args(argv)

    import torch

    from ..model.dialect import DIALECT
    from ..model.heads import PortHeads
    from ..model.lfm2_port import (
        DEFAULT_BASE, DEFAULT_REVISION, ControlTokens, PortBackbone, load_backbone, resolve_base,
    )
    from .gguf import export_heads_gguf, export_model_gguf, export_model_hf, fork_root

    out = Path(args.out)
    # Never replace a previous export, including an interrupted one.
    out.mkdir(parents=True, exist_ok=False)
    # Export needs model tensors, not the checkpoint's optimizer allocations.
    checkpoint_path = Path(args.heads).expanduser() if args.heads else None
    checkpoint_stat = _stat_identity(checkpoint_path) if checkpoint_path else None
    checkpoint_identity = _file_identity(checkpoint_path) if checkpoint_path else None
    state = torch.load(args.heads, map_location="cpu", weights_only=False, mmap=True) if args.heads else None
    metadata = (state or {}).get("port_config", {})
    if metadata.get('profile', 'legacy-rms-v1') != 'legacy-rms-v1':
        raise ValueError('raw-token-v1 requires its qualified PyTorch handoff; the legacy projector exporter is not compatible')
    cutoff = args.cutoff if args.cutoff is not None else metadata.get("cutoff")
    if state is not None and cutoff is None:
        raise ValueError("Legacy checkpoint has no cutoff metadata; supply its actual --cutoff")
    cutoff = cutoff if cutoff is not None else 6
    saved_length = int(state["heads"]["stop.position.weight"].shape[0]) - 1 if state else None
    max_block = args.max_block if args.max_block is not None else saved_length or 64
    if saved_length is not None and max_block != saved_length:
        raise ValueError(f"--max-block must match checkpoint length {saved_length}")
    if metadata.get("cutoff") is not None and cutoff != metadata["cutoff"]:
        raise ValueError("--cutoff differs from the trained checkpoint")
    torch.manual_seed(0)
    # Record the same resolution policy as load_backbone, while passing the original
    # arguments through unchanged so loading and merge behavior stay identical.
    resolved_base = resolve_base(args.base)
    crisp_lora_identity = _local_artifact_identity(args.lora)
    model, tokenizer = load_backbone(args.base, args.lora, dtype=torch.float32, device="cpu")
    backbone = PortBackbone(model, ControlTokens.from_tokenizer(tokenizer))
    heads = PortHeads(backbone, cutoff=cutoff, max_length=max_block).eval()
    if state is not None:
        heads.load_state_dict(state["heads"])
        with torch.no_grad():
            backbone.control_rows.copy_(state["control_rows"].to(backbone.control_rows))
        if state.get("lora"):
            from ..train.adapters import inject_lora, lora_state
            from peft.tuners.tuners_utils import BaseTunerLayer

            inject_lora(backbone, state["lora_layers"], rank=state["lora_rank"],
                        alpha=metadata.get("lora_alpha", 2 * state["lora_rank"]))
            if set(lora_state(backbone)) != set(state["lora"]):
                raise ValueError("Checkpoint backbone adapter names do not match the model")
            parameters = dict(backbone.hf.named_parameters())
            with torch.no_grad():
                for name, value in state["lora"].items():
                    parameters[name].copy_(value.to(parameters[name]))
            # Export ordinary merged HF weights, not unrecognised PEFT tensor names.
            for name, module in list(backbone.hf.named_modules()):
                if isinstance(module, BaseTunerLayer):
                    module.merge(safe_merge=True)
                    parent, _, child = name.rpartition(".")
                    setattr(backbone.hf.get_submodule(parent), child, module.get_base_layer())
    fork = fork_root()
    converter = fork / "convert_hf_to_gguf.py"
    converter_identity = _file_identity(converter) if converter.is_file() else None
    commit_result = subprocess.run(
        ["git", "-C", str(fork), "rev-parse", "--verify", "HEAD"],
        check=False, capture_output=True, text=True,
    )
    fork_commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    with tempfile.TemporaryDirectory() as tmp:
        hf_dir = export_model_hf(backbone, tokenizer, Path(tmp) / "hf")
        model_gguf = export_model_gguf(hf_dir, out / "model.gguf", outtype=args.outtype)
    heads_gguf = export_heads_gguf(heads, backbone, out / "neuralese.gguf", dialect=args.dialect or DIALECT)
    base_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    tokenizer_revision = (getattr(tokenizer, "init_kwargs", None) or {}).get("_commit_hash")
    default_revision_used = (DEFAULT_REVISION if resolved_base == DEFAULT_BASE else
                             DEFAULT_REVISION if Path(resolved_base).name == DEFAULT_REVISION else None)
    model_identity = _file_identity(model_gguf)
    heads_identity = _file_identity(heads_gguf)
    if checkpoint_path and _stat_identity(checkpoint_path) != checkpoint_stat:
        raise ValueError("Trainer checkpoint changed during export; preserve outputs and export a frozen snapshot")
    if _local_artifact_identity(args.lora) != crisp_lora_identity:
        raise ValueError("Crisp adapter changed during export; preserve outputs and export a frozen snapshot")
    if converter_identity and _sha256_file(converter) != converter_identity["sha256"]:
        raise ValueError("GGUF converter changed during export; preserve outputs for review")
    receipt = {
        "schema": "natlang-neuralese-export/2",
        "model": str(model_gguf),
        "heads": str(heads_gguf),
        "cutoff": heads.cutoff,
        "max_block": heads.max_length,
        "checkpoint": args.heads,
        "backbone_adapters_merged": bool((state or {}).get("lora")),
        "artifacts": {"model_gguf": model_identity, "heads_gguf": heads_identity},
        "trainer_checkpoint": checkpoint_identity,
        "base": {
            "requested": args.base,
            "resolved_source": resolved_base,
            "default_revision": default_revision_used,
            "model_config_commit": base_revision,
            "tokenizer_commit": tokenizer_revision,
            "training_base_identity": "not recorded in trainer checkpoint; export source is not proof of training source"
            if state is not None else None,
        },
        "crisp_lora": crisp_lora_identity,
        "training_adapters": {
            "merged_from_checkpoint": bool((state or {}).get("lora")),
            "layers": (state or {}).get("lora_layers", []),
            "rank": (state or {}).get("lora_rank"),
            "alpha": metadata.get("lora_alpha"),
        },
        "export": {
            "gguf_outtype": args.outtype,
            "backbone_load_dtype": "torch.float32",
            "cutoff": heads.cutoff,
            "max_block": heads.max_length,
            "dialect": args.dialect or DIALECT,
        },
        "converter": {
            "fork_root": str(fork),
            "git_commit": fork_commit,
            "converter": converter_identity,
        },
        "parity_claim": False,
    }
    (out / "export.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
