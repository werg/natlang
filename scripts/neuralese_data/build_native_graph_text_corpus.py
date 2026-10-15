#!/usr/bin/env python3
"""Stream an admitted, token-aligned text twin into native one-node graphs.

This is an explicit static transformation: ordinary full trajectories remain
ordinary one-node invocations. It does not infer role edges from visible text.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from transformers import AutoTokenizer
from natlang_neuralese.common.hashing import sha256_file_hex as sha256

from natlang_neuralese.data.native_graph_conversion import text_row_graph, whole_trajectory_text_row_graph
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
from natlang_neuralese.model.shared_state import ROLE_MARKERS


def read_jsonl_zstd(path):
    proc = subprocess.Popen(['zstd', '-dcq', str(path)], stdout=subprocess.PIPE)
    try:
        for number, raw in enumerate(proc.stdout, 1):
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{number}: invalid JSON') from exc
        code = proc.wait()
        if code:
            raise RuntimeError(f'zstd decompression failed with status {code}: {path}')
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait()


class GraphWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self.temporary = tempfile.mkstemp(prefix='.' + self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        os.close(fd)
        self.out = open(self.temporary, 'wb')
        self.encoder = subprocess.Popen(['zstd', '-T2', '-5', '-c'], stdin=subprocess.PIPE, stdout=self.out)
        self.rows = 0

    def write(self, graph):
        self.encoder.stdin.write(json.dumps(graph, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n')
        self.rows += 1

    def finish(self):
        self.encoder.stdin.close()
        code = self.encoder.wait()
        self.out.flush()
        os.fsync(self.out.fileno())
        self.out.close()
        if code:
            Path(self.temporary).unlink(missing_ok=True)
            raise RuntimeError(f'zstd compression failed for {self.path}: {code}')
        os.replace(self.temporary, self.path)
        return {'path': str(self.path), 'rows': self.rows, 'bytes': self.path.stat().st_size,
                'sha256': sha256(self.path)}

    def abort(self):
        if self.encoder.poll() is None:
            try:
                self.encoder.stdin.close()
            except Exception:
                pass
            self.encoder.terminate()
            self.encoder.wait()
        if not self.out.closed:
            self.out.close()
        Path(self.temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--base-tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-id', required=True)
    parser.add_argument('--context-weight', type=float, default=.05)
    parser.add_argument('--source-sha256', required=True)
    args = parser.parse_args()
    if sha256(args.source) != args.source_sha256:
        raise ValueError('source corpus compressed SHA differs from the admitted binding')
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_tokenizer), trust_remote_code=False,
                                               local_files_only=True)
    source_fingerprint = tokenizer_fingerprint(tokenizer)
    if not hasattr(tokenizer, 'backend_tokenizer'):
        raise TypeError('source tokenizer lacks the fast backend required by the model loader')
    try:
        import inspect
        modern = 'replace_extra_special_tokens' in inspect.signature(tokenizer.add_special_tokens).parameters
        key = 'extra_special_tokens' if modern else 'additional_special_tokens'
        option = 'replace_extra_special_tokens' if modern else 'replace_additional_special_tokens'
        tokenizer.add_special_tokens({key: list(ROLE_MARKERS)}, **{option: False})
    except Exception as exc:
        raise RuntimeError('could not provision the same four native role tokens as the backbone loader') from exc
    role_ids = tuple(tokenizer.convert_tokens_to_ids(marker) for marker in ROLE_MARKERS)
    if len(set(role_ids)) != 4 or any(token is None or token == tokenizer.unk_token_id for token in role_ids):
        raise ValueError('expanded tokenizer did not yield four distinct role token IDs')
    expanded_fingerprint = tokenizer_fingerprint(tokenizer)
    rows = read_jsonl_zstd(args.source)
    # Stream each row directly to one of two compressed outputs; the source is
    # never materialized and held rows remain a separate trainer input.
    writers = {split: GraphWriter(args.output / split / 'native-graphs.jsonl.zst')
               for split in ('train', 'test')}
    train_groups, held_groups = set(), set()
    whole_policy = {split: {'rows': 0, 'failed_assistant_tokens_masked': 0,
                            'runtime_opening_assistant_turns': 0,
                            'runtime_opening_assistant_tokens': 0,
                            'assistant_target_tokens': 0}
                    for split in ('train', 'test')}
    total = 0
    try:
        for row in rows:
            split = row.get('split')
            if split not in {'train', 'test'}:
                raise ValueError(f'admitted row has invalid split {split!r}')
            convert = (whole_trajectory_text_row_graph
                       if row.get('supervision') == 'whole-trajectory' else text_row_graph)
            graph = convert(row, source_tokenizer_fingerprint=source_fingerprint,
                tokenizer=tokenizer, expanded_tokenizer_fingerprint=expanded_fingerprint,
                **({'context_weight': args.context_weight, 'feedback_weight': 0.25}
                   if convert is whole_trajectory_text_row_graph
                   else {'context_weight': args.context_weight}))
            writer = writers[split]
            groups = train_groups if split == 'train' else held_groups
            groups.update(graph['source_groups'])
            if convert is whole_trajectory_text_row_graph:
                policy = graph['conversion']['supervision_policy']
                tally = whole_policy[split]
                tally['rows'] += 1
                for key in ('failed_assistant_tokens_masked', 'runtime_opening_assistant_turns',
                            'runtime_opening_assistant_tokens', 'assistant_target_tokens'):
                    tally[key] += policy[key]
            writer.write(graph)
            total += 1
        overlap = train_groups & held_groups
        if overlap:
            raise ValueError('train/held graph groups overlap: ' + ', '.join(sorted(overlap)[:12]))
        summaries = [writers[split].finish() for split in ('train', 'test')]
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    receipt = {'schema': 'natlang.native-graph-text-build/1', 'model_id': args.model_id,
        'source': {'path': str(args.source), 'sha256': args.source_sha256, 'rows': total},
        'source_tokenizer_fingerprint': source_fingerprint,
        'expanded_tokenizer_fingerprint': expanded_fingerprint,
        'role_token_ids': list(role_ids), 'context_weight': args.context_weight,
        'outputs': summaries,
        'conversion': ('static one-node graphs; whole-trajectory rows preserve all roles, weight assistant targets, '
                       'downweight prompt/runtime-opening/tool context, and exclude failed assistant actions; '
                       'no inferred native edges'),
        'whole_trajectory_supervision': whole_policy if any(row['rows'] for row in whole_policy.values()) else None}
    args.output.mkdir(parents=True, exist_ok=True)
    temp_receipt = args.output / '.receipt.json.tmp'
    temp_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    with temp_receipt.open('rb') as handle:
        os.fsync(handle.fileno())
    os.replace(temp_receipt, args.output / 'receipt.json')
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
