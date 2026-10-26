"""Export each distinct effective stack of a release as one PEFT LoRA, from release files alone (plan §7).

Backends that apply one adapter per request (vLLM) serve a call's whole stack as one exported LoRA, named by the
runtime's canonical stack identity. The sum is exact: members slicing one canonical basis add in that basis (rank at
most the largest member's), everything else concatenates (`stack.stack_factors`). No engine is loaded; coefficients
and bases come from the release's `artifacts/*.nnt`.

`python -m natlang_neuralese.tune.stack_export --release DIR --stacks stacks.json --module-map lfm2 --hf-config DIR
--out DIR` where stacks.json is ``[{"identity", "entries": [{"id", "scale", "node"}]}]`` (the runtime's resolved
effective stacks). Writes ``<out>/<identity>/`` PEFT directories and ``<out>/stack-exports.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import TuneMathError
from .bases import CanonicalBasis, tensor_digest
from .export import export_peft
from .parameterization import FreeLoRADelta, SubspaceDelta
from .stack import StackMember, stack_factors
from .targets import module_map


def load_artifacts(directory):
    from ..model.native_artifact import decode_native_tensor_artifact
    artifacts = {}
    for path in sorted(Path(directory).glob('*.nnt')):
        artifact, ident = decode_native_tensor_artifact(path.read_bytes())
        if path.stem != ident:
            raise TuneMathError(f'artifact file {path.name} does not hold its named content')
        artifacts[ident] = artifact
    return artifacts


def members_from_artifacts(artifacts, entries):
    """`StackMember`s and basis views for runtime stack entries (artifact ID, scale, node)."""
    members, bases, shapes = [], {}, {}
    for entry in entries:
        record = artifacts.get(entry['id'])
        if record is None or record.kind != 'adapter-coefficients':
            raise TuneMathError(f"stack entry {entry['id']} is not a coefficient artifact of this release")
        spec = record.provenance['adapter_spec']
        deltas = {}
        for name in spec['parameter_names']:
            if spec['kind'] == 'free-lora':
                B, A = record.tensors['lora_B.'+name], record.tensors['lora_A.'+name]
                deltas[name] = FreeLoRADelta(B, A, spec['alpha'])
                shapes[name] = (*B.shape[:-1], A.shape[-1])
                continue
            basis = artifacts.get(record.source_artifact_id)
            if basis is None:
                raise TuneMathError(f"coefficients {entry['id']} lack their basis artifact in this release")
            key = f'{record.source_artifact_id}:{name}'
            left, right = basis.tensors['left.'+name], basis.tensors['right.'+name]
            if key not in bases:
                bases[key] = CanonicalBasis(key, name, left.shape[-1], left, right, tensor_digest(left, right))
            rank = spec['rank']
            core = record.tensors[name]
            deltas[name] = SubspaceDelta(key, rank, core.reshape(*core.shape[:-1], rank, rank))
            shapes[name] = (*left.shape[:-1], right.shape[-2])
        level = entry['node'].split(':', 1)[0]  # family:, tree:, folder:, lambda: scope node IDs
        members.append(StackMember(entry['node'], level, float(entry['scale']), deltas))
    return members, bases, shapes


def export_stacks(release, stacks, out, *, family, hf_config, base_identity):
    artifacts = load_artifacts(Path(release) / 'artifacts')
    out = Path(out)
    exported = []
    for stack in stacks:
        identity = stack['identity']
        if not stack['entries']:
            continue  # the base needs no adapter
        target = out / identity
        members, bases, shapes = members_from_artifacts(artifacts, stack['entries'])
        targets, _ = module_map(sorted(shapes), family=family, classes=None)
        if not (target / 'natlang-export-receipt.json').exists():
            export_peft(stack_factors(members, bases, shapes), targets, target, base_identity=base_identity,
                        base_model_name_or_path=str(hf_config), stack_identity=identity)
        receipt = json.loads((target / 'natlang-export-receipt.json').read_text())
        exported.append({'identity': identity, 'directory': identity, 'receipt_sha256': receipt['receipt_sha256'],
                         'ranks': receipt['ranks']})
    (out / 'stack-exports.json').write_text(json.dumps(exported, indent=1, sort_keys=True))
    return exported


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name in ('--release', '--stacks', '--module-map', '--hf-config', '--out'):
        parser.add_argument(name, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads((Path(args.release) / 'natlang.adapters.json').read_text())
    export_stacks(args.release, json.loads(Path(args.stacks).read_text()), args.out, family=args.module_map,
                  hf_config=args.hf_config, base_identity=manifest['base'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
