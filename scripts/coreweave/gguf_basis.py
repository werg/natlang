#!/usr/bin/env python3
"""A base-model basis in a trained GGUF's byte layout, so a trained GGUF crosses the slow DGX uplink as a small
XOR delta (scripts/coreweave/lossless_file_transfer.py) instead of 3.5 GB.

A trained LFM3 export and the official base GGUF differ in header length and in a few extra tensors, so their bytes
do not line up. The basis is a file of the trained GGUF's exact size holding, at each tensor's data offset, the base
GGUF's tensor of the same name and size, and zeros everywhere else (header, padding, tensors the base lacks). The
XOR of a trained GGUF against it is mostly zero bits (measured 2026-10-30: 0.7-50 % of the raw size after zstd,
versus 78 % for the raw GGUF).

  gguf_basis.py layout BASE_GGUF TRAINED_GGUF OUT.json   (DGX; needs gguf-py from the pinned converter snapshot)
  gguf_basis.py build LAYOUT.json BASE_GGUF OUT           (anywhere; stdlib only; verifies the basis identity)

The layout binds the base identity, the copy list and the resulting basis identity; ``build`` refuses a base or a
result with a different sha256, so the delta is decoded only against the exact intended basis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

SCHEMA = 'natlang.gguf-basis-layout/1'
BLOCK = 8 << 20


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(BLOCK), b''):
            digest.update(block)
    return digest.hexdigest()


def write_basis(layout, base, out):
    """Write the basis described by `layout` from `base` into `out`; returns its sha256."""
    digest = hashlib.sha256()
    position = 0
    with open(base, 'rb') as source, open(out, 'wb') as sink:
        def emit(data):
            nonlocal position
            sink.write(data)
            digest.update(data)
            position += len(data)

        def zeros(count):
            while count:
                chunk = min(count, BLOCK)
                emit(bytes(chunk))
                count -= chunk

        for base_offset, target_offset, size in layout['copies']:
            if target_offset < position:
                raise ValueError('layout copies overlap or are unsorted')
            zeros(target_offset - position)
            source.seek(base_offset)
            remaining = size
            while remaining:
                chunk = source.read(min(remaining, BLOCK))
                if not chunk:
                    raise ValueError('base GGUF is shorter than the layout')
                emit(chunk)
                remaining -= len(chunk)
        zeros(layout['basis']['bytes'] - position)
        sink.flush()
        os.fsync(sink.fileno())
    return digest.hexdigest()


def make_layout(base, trained):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.development-source-snapshots' /
                           'llama.cpp-a6401bbd' / 'gguf-py'))
    from gguf import GGUFReader
    base_tensors = {tensor.name: tensor for tensor in GGUFReader(base).tensors}
    copies, missing = [], []
    for tensor in sorted(GGUFReader(trained).tensors, key=lambda t: t.data_offset):
        match = base_tensors.get(tensor.name)
        if match is None or match.n_bytes != tensor.n_bytes:
            missing.append(tensor.name)
            continue
        copies.append([int(match.data_offset), int(tensor.data_offset), int(tensor.n_bytes)])
    layout = {'schema': SCHEMA, 'base': {'sha256': sha256_file(base), 'bytes': os.path.getsize(base)},
              'copies': copies, 'zero_tensors': missing,
              'basis': {'sha256': None, 'bytes': os.path.getsize(trained)}}
    return layout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('layout')
    p.add_argument('base'), p.add_argument('trained'), p.add_argument('out')
    p.add_argument('--basis-out', help='also keep the built basis here (else it is built in a temp file and removed)')
    p = sub.add_parser('build')
    p.add_argument('layout'), p.add_argument('base'), p.add_argument('out')
    args = parser.parse_args(argv)
    if args.command == 'layout':
        layout = make_layout(args.base, args.trained)
        basis = args.basis_out or f'{args.out}.basis.tmp'
        layout['basis']['sha256'] = write_basis(layout, args.base, basis)
        if not args.basis_out:
            os.unlink(basis)
        Path(args.out).write_text(json.dumps(layout) + '\n')
        print(json.dumps({'basis': layout['basis'], 'copies': len(layout['copies']),
                          'zero_tensors': layout['zero_tensors']}))
    else:
        layout = json.loads(Path(args.layout).read_text())
        if layout.get('schema') != SCHEMA:
            raise SystemExit('not a GGUF basis layout')
        base = {'sha256': sha256_file(args.base), 'bytes': os.path.getsize(args.base)}
        if base != layout['base']:
            raise SystemExit(f'base GGUF identity {base} differs from the layout {layout["base"]}')
        pending = f'{args.out}.partial'
        if write_basis(layout, args.base, pending) != layout['basis']['sha256']:
            os.unlink(pending)
            raise SystemExit('built basis differs from the layout identity')
        os.replace(pending, args.out)
        print(json.dumps({'basis': layout['basis']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
