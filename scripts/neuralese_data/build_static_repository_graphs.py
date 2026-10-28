"""Pinned complete repository files through the shared ordinary text graph law."""
from __future__ import annotations
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def source_row(value, *, source_sha256, position):
    """Verify original repository headers and code body; never repair source text."""
    from scripts.neuralese_data.records import group_key
    owner, repo, path, commit, text = (value.get(k) for k in
                                     ('owner', 'repo', 'relpath', 'commit_sha', 'text'))
    if (not all(isinstance(x, str) and x for x in (owner, repo, path, commit, text))
            or any(c in owner + repo + path for c in ('\n', '\r', '\x00'))
            or '/' in owner or '/' in repo or not re.fullmatch('[0-9a-f]{40}', commit)
            or not re.fullmatch('[0-9a-f]{64}', str(value.get('sha256', '')))
            or type(value.get('size_bytes')) is not int or value['size_bytes'] < 0):
        raise ValueError('invalid original repository identity or code metadata')
    header = f'<REPO_NAME>{owner}/{repo} <FILE_PATH>{path}\n'
    if not text.startswith(header):
        raise ValueError('original repository/path header differs from source fields')
    body = text[len(header):].encode('utf-8')
    raw_sha = hashlib.sha256(body).hexdigest()
    if len(body) != value['size_bytes']:
        raise ValueError('original complete code body differs from upstream size/SHA')
    digest_law = 'original-utf8-bytes'
    if raw_sha != value['sha256']:
        # This source records original CRLF byte size but an LF-normalized SHA.
        # Authenticate that exact full-body law without changing target bytes.
        if b'\r\n' not in body or hashlib.sha256(body.replace(b'\r\n', b'\n')).hexdigest() != value['sha256']:
            raise ValueError('original complete code body differs from upstream size/SHA')
        digest_law = 'original-size-and-crlf-to-lf-digest'
    if not body:
        raise ValueError('empty original file has no code continuation')
    # Repo keys use the same shared namespace as existing acquired SWE sources.
    groups = [group_key('repo', owner + '/' + repo), 'source-content-sha256:' + value['sha256']]
    return dict(id=f'static-code:{source_sha256}:{position}', text=text,
                source_groups=groups, provenance=dict(source_row=position,
                repository=owner+'/'+repo, path=path, commit_sha=commit,
                code_sha256=raw_sha, upstream_code_sha256=value['sha256'],
                upstream_digest_law=digest_law, code_bytes=value['size_bytes']))


def disposition(groups, protection):
    """Apply existing closed protected sets; quality/pending never become targets."""
    keys = set(groups)
    if keys & (protection['quality'] | protection['pending']):
        return 'pending'
    return 'test' if keys & protection['held'] else 'train'


def continuation_graph(row, split, source_tokenizer, expanded_tokenizer, source_fp, expanded_fp):
    """Tokenize with the original vocabulary; expanded roles only validate decoding."""
    from natlang_neuralese.data.native_graph_conversion import text_row_graph
    ids = source_tokenizer(row['text'], add_special_tokens=False, truncation=False)['input_ids']
    if not ids:
        raise ValueError('nonempty complete source file tokenized empty')
    return text_row_graph({**row, 'token_ids': ids, 'tokenizer_sha256': source_fp,
        'supervised_suffix_start': 0, 'split': split}, tokenizer=expanded_tokenizer,
        source_tokenizer_fingerprint=source_fp, expanded_tokenizer_fingerprint=expanded_fp)


def main():
    from natlang_neuralese.common import line_backbones
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--expected-rows', type=int, required=True)
    parser.add_argument('--acquisition', type=Path, required=True)
    parser.add_argument('--acquisition-sha256', required=True)
    parser.add_argument('--protection-manifest', type=Path, required=True)
    parser.add_argument('--base-tokenizer', type=Path, required=True)
    parser.add_argument('--model-key', choices=tuple(line_backbones.BACKBONES), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from natlang_neuralese.common.hashing import sha256_file_hex as sha, canonical_json_sha256_hex
    from natlang_neuralese.common.jsonio import write_json_durable
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
    from scripts.neuralese_data.splits import close_group_sets
    from scripts.neuralese_data.conversion_output_lease import acquire_output_lease

    def binding(path):
        return dict(path=str(path.resolve()), sha256=sha(path), bytes=path.stat().st_size)
    source = binding(args.source)
    acquisition = binding(args.acquisition)
    if source['sha256'] != args.source_sha256 or acquisition['sha256'] != args.acquisition_sha256:
        raise ValueError('static source acquisition binding changed')
    acquired = json.loads(args.acquisition.read_text())
    if acquired['ref'] != 'D39' or acquired['source'] != source or acquired['training_admission'] is not False:
        raise ValueError('static source differs from exact non-admission acquisition')
    if args.expected_rows != acquired['parquet_rows']:
        raise ValueError('static source row obligation differs from acquisition')
    parquet = pq.ParquetFile(args.source)
    if parquet.metadata.num_rows != args.expected_rows:
        raise ValueError('Parquet row count differs from acquisition')

    def rows():
        position = 0
        for batch in parquet.iter_batches(batch_size=16):
            for value in batch.to_pylist():
                yield position, value
                position += 1

    with acquire_output_lease(args.output):
        if (args.output/'receipt.json').exists():
            raise ValueError('completed static conversion is immutable')
        protection = current_protection(args.protection_manifest)
        groups = []
        invalid = {}
        for position, value in rows():
            if position % 4096 == 0:
                print(json.dumps({'phase': 'source_metadata', 'source_row': position}), flush=True)
            try:
                row = source_row(value, source_sha256=source['sha256'], position=position)
                groups.append(row['source_groups'])
            except ValueError as error:
                invalid[position] = str(error)
        closed, _ = close_group_sets(groups, {k: protection[k] for k in ('held', 'quality', 'pending')})
        del groups
        tokenizer = AutoTokenizer.from_pretrained(str(args.base_tokenizer), local_files_only=True,
                                                  trust_remote_code=False)
        original_fp = tokenizer_fingerprint(tokenizer)
        expanded_tokenizer = copy.deepcopy(tokenizer)
        roles = provision(expanded_tokenizer)
        expanded_fp = tokenizer_fingerprint(expanded_tokenizer)
        writers = {split: GraphWriter(args.output/split/'native-graphs.jsonl.zst') for split in ('train', 'test')}
        counts = Counter()
        pending = args.output/'source-pending.jsonl'
        provenance = args.output/'source-row-proofs.jsonl'
        seen = set()
        try:
            with pending.open('x') as omitted, provenance.open('x') as proof:
                for position, value in rows():
                    if position % 512 == 0:
                        print(json.dumps({'phase': 'static_code_conversion', 'model': args.model_key,
                                          'source_row': position, 'counts': dict(counts)}), flush=True)
                    if position in invalid:
                        omitted.write(json.dumps(dict(source_row=position, reason=invalid[position],
                                      source_row_sha256=canonical_json_sha256_hex(value)))+'\n')
                        counts['invalid_source_pending'] += 1
                        continue
                    row = source_row(value, source_sha256=source['sha256'], position=position)
                    split = disposition(row['source_groups'], closed)
                    if split == 'pending':
                        omitted.write(json.dumps(dict(source_row=position, reason='protected-quality-or-pending',
                                      source_groups=row['source_groups']))+'\n')
                        counts['protected_source_pending'] += 1
                        continue
                    # An identical raw code body has one representative; all original headers remain accounted.
                    identity = (split, row['provenance']['code_sha256'])
                    if identity in seen:
                        proof.write(json.dumps(dict(source_row=position, upstream_code_sha256=value['sha256'],
                                    code_sha256=row['provenance']['code_sha256'],
                                    disposition='represented-identical-raw-code-body', split=split))+'\n')
                        counts['represented_code_digest'] += 1
                        continue
                    seen.add(identity)
                    graph = continuation_graph(row, split, tokenizer, expanded_tokenizer,
                                               original_fp, expanded_fp)
                    tokens = len(graph['nodes'][0]['token_ids'])
                    graph['conversion']['static_repository_source'] = {**row['provenance'],
                        'parquet': source, 'acquisition': acquisition, 'original_row_sha256': canonical_json_sha256_hex(value),
                        'target_origin': 'original complete code text; static next-token continuation',
                        'execution_or_answer_success_asserted': False}
                    writers[split].write(graph)
                    counts[split] += 1
                    proof.write(json.dumps(dict(source_row=position, id=graph['id'], split=split,
                        upstream_code_sha256=value['sha256'], code_sha256=row['provenance']['code_sha256'], graph_sha256=canonical_json_sha256_hex(graph),
                        tokens=tokens, provenance=row['provenance']))+'\n')
            if sum(counts.values()) != args.expected_rows:
                raise ValueError('static conversion lost original source row accounting')
            outputs = {split: writers[split].finish() for split in writers}
        except BaseException:
            for writer in writers.values():
                writer.abort()
            raise
        if binding(args.source) != source or binding(args.acquisition) != acquisition:
            raise ValueError('static source changed during complete conversion')
        if current_protection(args.protection_manifest)['proof'] != protection['proof']:
            raise ValueError('static source protection changed during conversion')
        write_json_durable(args.output/'receipt.json', dict(schema='natlang.static-repository-native-build/1',
            source=source, acquisition=acquisition, source_rows=args.expected_rows, model_key=args.model_key,
            converter=binding(Path(__file__)), tokenizer={'source_fingerprint': original_fp,
                'expanded_fingerprint': expanded_fp, 'role_token_ids': list(roles)},
            counts=dict(counts), outputs=outputs, current_protection=protection['proof'],
            source_pending=binding(pending), row_proofs=binding(provenance),
            conversion='static one-node original code continuation; no assistant answer or execution assertion',
            whole_files=True, windowed=False, training_admission=False,
            communication_foundation_qualification=False, training_exposure=0))
        print(json.dumps({'passed': True, 'counts': dict(counts), 'output': str(args.output),
                          'training_admission': False}), flush=True)


if __name__ == '__main__':
    main()
