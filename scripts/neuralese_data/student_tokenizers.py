#!/usr/bin/env python3
"""The pinned student tokenizers that rendered corpora and tokenizer twins are produced for (one table).

Every render/twin wrapper takes its ``--tokenizer NAME=DIR`` arguments from here instead of hard-coding the LFM2.5
snapshot, so adding a student (LFM3-1.7B, 2026-10-28) is one row. A row binds the tokenizer directory to the exact
``tokenizer.json`` / ``chat_template.jinja`` bytes and the backbone's declared ``model_type`` (its tool-call format
and history-reasoning policy come from ``natlang_neuralese.serve.chat``), and names the model id prefix of its native
graph views (``build_native_graph_text_corpus.py --model-id``: ``<architecture>:<base inventory sha256>``, the id
``native_model.load_native_model`` gives the loaded base).

``DEFAULT_STUDENTS`` stays the production pair until the LFM3 cutover (plans/neuralese/LFM3_PORT_20261028.md, part
D); an LFM3 render is requested explicitly (``STUDENTS="lfm3"`` or ``lfm25 mellum21 lfm3`` in the wrappers).

  student_tokenizers.py args NAME...     the --tokenizer NAME=DIR arguments (pins verified)
  student_tokenizers.py mounts NAME...   docker -v arguments for directories outside the repo and data mounts
  student_tokenizers.py path NAME        the tokenizer directory
  student_tokenizers.py model-id NAME    the native graph model id (neuralese venv; hashes the base weights through the shared
                                         verified file cache)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve as machine_path  # noqa: E402 (stdlib-only module)

REPO = machine_path('repo')
HF_CACHE = Path.home() / '.cache/huggingface/hub'
STUDENTS = {
    'lfm25': {'path': REPO / 'models/hf/hub/models--LiquidAI--LFM2.5-350M/snapshots/9e6c6ccf47cd318696e137d381a7ded8fe4df09f',
              'model_type': 'lfm2',
              'model_id': 'lfm2:f40d7a5bf651a03618ae8c06bda24a180e22fa79de964e55862c2c4984249da6'},
    'mellum21': {'path': machine_path('models', 'mellum21-12b-a2.5b-thinking'), 'model_type': 'mellum',
                 'model_id': 'mellum:bc700045baee1c7ea1869675c38efedd59d9db09975ea0b38cff8db152d6148f'},
    'lfm3': {'path': HF_CACHE / 'models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd',
             # Snapshot files are symlinks into the repository's blobs: mount the whole model repository.
             'mount': HF_CACHE / 'models--LiquidAI--LFM3-1.7B', 'model_type': 'lfm3',
             'tokenizer_sha256': '7e226e34bface23ccf6e495d157bf77186c1c30a7659880e1a8994db528c3701',
             'chat_template_sha256': 'f45005c50cd2a97329722bf3a2dee9f16c04cdf825a589b280fb7e99e6827818'},
}
DEFAULT_STUDENTS = ('lfm25', 'mellum21')
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'special_tokens_map.json',
                   'config.json', 'generation_config.json')
MOUNTED = (REPO,machine_path('data_nvme'), machine_path('data_hdd'))  # the wrappers' standard read-only mounts


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(name: str) -> dict:
    """The verified row of a student: its pinned files must match their recorded digests."""
    if name not in STUDENTS:
        raise SystemExit(f'unknown student tokenizer {name!r} (declared: {sorted(STUDENTS)})')
    row = STUDENTS[name]
    for key, filename in (('tokenizer_sha256', 'tokenizer.json'), ('chat_template_sha256', 'chat_template.jinja')):
        if key in row and _sha(row['path'] / filename) != row[key]:
            raise SystemExit(f'{name}: {filename} differs from its pinned sha256 {row[key]}')
    if not (row['path'] / 'tokenizer.json').is_file():
        raise SystemExit(f'{name}: no tokenizer.json in {row["path"]}')
    return row


def model_id(name: str) -> str:
    """`<architecture>:<base inventory sha256>` of the student's base, as native_model.load_native_model names it."""
    row = resolve(name)
    if 'model_id' in row:
        return row['model_id']
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex
    from natlang_neuralese.model.native_model import base_inventory
    return f"{row['model_type']}:{canonical_json_sha256_hex(base_inventory(row['path']))}"


def main(argv):
    if len(argv) < 2 or argv[0] not in ('args', 'mounts', 'path', 'model-id'):
        raise SystemExit(__doc__)
    command, names = argv[0], argv[1:]
    rows = [(name, resolve(name)) for name in names]
    if command == 'args':
        print(' '.join(f'--tokenizer {name}={row["path"]}' for name, row in rows))
    elif command == 'mounts':
        # The model repository, plus the real directories its tokenizer files resolve to: HF snapshot files are
        # symlinks into the repository's blobs, which may themselves link into a shared hub blob store
        # (~/.cache/huggingface/hub/blobs/xx/, LFM3 2026-10-28); a mount of the repository alone leaves them dangling.
        candidates = set()
        for _, row in rows:
            candidates.add(Path(row.get('mount', row['path'])))
            for filename in TOKENIZER_FILES:
                if (row['path'] / filename).exists():
                    candidates.add((row['path'] / filename).resolve(strict=True).parent)
        mounts = {str(path) for path in candidates
                  if not any(path.is_relative_to(root) for root in MOUNTED)
                  and not any(path != other and path.is_relative_to(other) for other in candidates)}
        print(' '.join(f'-v {path}:{path}:ro' for path in sorted(mounts)))
    elif command == 'path':
        print(rows[0][1]['path'])
    else:
        print(model_id(names[0]))


if __name__ == '__main__':
    main(sys.argv[1:])
