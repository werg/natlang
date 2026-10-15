"""Prepare a complete immutable shared-state snapshot for the existing private exporter.

The exporter uploads the returned archive, verifies a full private GET, then
releases this snapshot's closure pin. No account credentials or training policy
belong here. A global manifest inode alone is never an exportable checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

try:
    from . import source_archive
except ImportError:  # Direct CLI in an isolated runtime.
    import source_archive


def prepare(checkpoint, spool, *, reserve_bytes=20 * 2**30):
    from natlang_neuralese.model.native_shared_checkpoint import pin_shared_checkpoint
    from natlang_neuralese.model.native_bundle import verify_bundle
    checkpoint, spool = Path(checkpoint).resolve(), Path(spool).resolve()
    pinned = pin_shared_checkpoint(checkpoint, consumer='private-periodic-export')  # Full recovery.
    digest = pinned['checkpoint']['sha256']
    pin = Path(pinned['checkpoint']['resolved']).parent
    name = 'native-shared-' + digest
    receipt_path = spool/(name + '.receipt.json')
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if Path(receipt['source']).resolve() != pin:
            raise ValueError('shared export receipt belongs to another closure pin')
    else:
        receipt = source_archive.pack(pin, spool, name, reserve_bytes=reserve_bytes)
    proof = source_archive.verify(spool, name)
    verified = verify_bundle(pin)
    if verified['participant'] is not None or verified['checkpoint']['sha256'] != digest:
        raise ValueError('shared durability requires the complete unchanged global checkpoint')
    result = {'schema': 'natlang.native-shared-private-export/1',
              'checkpoint': {'path': str(checkpoint), 'sha256': digest},
              'pin': str(pin), 'pin_consumer': 'private-periodic-export', 'spool': str(spool), 'name': name,
              'archive': {'path': receipt['archive'], 'sha256': receipt['archive_sha256'],
                          'bytes': receipt['archive_bytes']},
              'closure_files': receipt['manifest']['files'],
              'roundtrip': proof, 'private_durability': 'pending'}
    source_archive._write_json(spool/(name + '.shared-export.json'), result)
    return result


def release(prepared, private_receipt):
    """Release only the assigned pin after upload AND independent full GET agree."""
    if prepared.get('schema') != 'natlang.native-shared-private-export/1':
        raise ValueError('shared export preparation schema differs')
    archive = prepared['archive']
    identity = private_receipt.get('identity', {})
    fetched = private_receipt.get('independent_verification', {})
    expected = {key: archive[key] for key in ('sha256', 'bytes')}
    if (not private_receipt.get('private') or private_receipt.get('source') != archive['path']
            or {key: identity.get(key) for key in expected} != expected
            or {key: fetched.get(key) for key in expected} != expected):
        raise ValueError('shared closure pin requires independently verified complete private archive')
    source_archive.verify(prepared['spool'], prepared['name'])
    checkpoint = Path(prepared['checkpoint']['path']).resolve()
    pin = Path(prepared['pin']).resolve()
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    if prepared.get('pin_consumer') != 'private-periodic-export':
        raise ValueError('shared export pin belongs to another consumer')
    scope = 'full-' + canonical_json_sha256_hex(prepared['pin_consumer'])
    expected_pin = checkpoint.parent/'pins'/prepared['checkpoint']['sha256']/scope
    if pin != expected_pin:
        raise ValueError('shared export cannot release an unrelated pin')
    if pin.exists():
        from natlang_neuralese.model.native_bundle import verify_bundle
        verified = verify_bundle(pin)
        if verified['checkpoint']['sha256'] != prepared['checkpoint']['sha256']:
            raise ValueError('shared export pin identity differs')
        shutil.rmtree(pin)
    result = {**prepared, 'private_durability': 'independent-full-get-sha-verified',
              'pin_released': True, 'private_receipt': private_receipt}
    source_archive._write_json(Path(prepared['spool'])/(prepared['name']+'.shared-export.json'), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    item = commands.add_parser('prepare')
    item.add_argument('--checkpoint', required=True)
    item.add_argument('--spool', required=True)
    item.add_argument('--reserve-gb', type=float, default=20)
    item = commands.add_parser('release')
    item.add_argument('--prepared', required=True)
    item.add_argument('--private-receipt', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.checkpoint, args.spool, reserve_bytes=int(args.reserve_gb*2**30))
    else:
        result = release(json.loads(Path(args.prepared).read_text()),
                         json.loads(Path(args.private_receipt).read_text()))
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
