"""Exact new-source replay for complete static repository continuations."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from scripts.neuralese_data.build_static_repository_graphs import source_row, disposition, continuation_graph
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease


def binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256_file_hex(path), bytes=path.stat().st_size)


def check(ref):
    if binding(ref['path']) != ref:
        raise ValueError('static admission evidence binding changed')
    return Path(ref['path'])


def json_rows(path):
    with open_jsonl_text(Path(path)) as stream:
        for line in stream:
            if not line.endswith('\n') or not line.strip():
                raise ValueError('static source evidence lost complete JSONL rows')
            yield json.loads(line)


def next_exact(stream, expected):
    if canonical_json_sha256_hex(next(stream, None)) != canonical_json_sha256_hex(expected):
        raise ValueError('static source graph, row proof or pending accounting differs from exact replay')


def verify_audit(audit_path):
    """Authenticate a saved all-row fidelity result before capacity partition."""
    audit_path = Path(audit_path).resolve(strict=True)
    audit = json.loads(audit_path.read_text())
    if (audit.get('schema') != 'natlang.static-repository-admission-audit/1'
            or audit.get('status') != 'passed' or audit.get('all_row_source_render_verified') is not True
            or audit.get('verifier') != binding(__file__)):
        raise ValueError('static admission requires the exact all-row verifier authority')
    for ref in audit['evidence']:
        check(ref)
    check(audit['row_proof'])
    return audit


def prepare(receipt_paths, tokenizers, protection_path, out):
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
    from scripts.neuralese_data.splits import close_group_sets
    from scripts.neuralese_data import build_static_repository_graphs as builder
    from natlang_neuralese.data import native_graph_conversion, text_corpus

    out = Path(out).resolve()
    protection = current_protection(Path(protection_path))
    evidence = {str(Path(__file__).resolve()): binding(__file__)}
    evidence[str(Path(protection_path).resolve())] = binding(protection_path)
    for module in (builder, native_graph_conversion, text_corpus):
        ref = binding(module.__file__); evidence[ref['path']] = ref
    receipts = {}
    for path in receipt_paths:
        ref = binding(path); evidence[ref['path']] = ref
        receipt = json.loads(Path(path).read_text()); model = receipt['model_key']
        if (model in receipts or receipt['schema'] != 'natlang.static-repository-native-build/1'
                or receipt['current_protection'] != protection['proof']
                or receipt['whole_files'] is not True or receipt['windowed'] is not False
                or receipt['training_admission'] is not False
                or receipt['converter'] != binding(builder.__file__)):
            raise ValueError('static conversion is not a distinct whole-source protected model view')
        for key in ('source', 'acquisition', 'converter', 'source_pending', 'row_proofs'):
            check(receipt[key]); evidence[receipt[key]['path']] = receipt[key]
        for ref in receipt['outputs'].values():
            physical = {k: ref[k] for k in ('path', 'sha256', 'bytes')}
            check(physical); evidence[physical['path']] = physical
        receipts[model] = receipt
    if set(receipts) != set(tokenizers) or len({r['source']['sha256'] for r in receipts.values()}) != 1:
        raise ValueError('static admission requires the same complete source in every declared tokenizer view')
    first = next(iter(receipts.values()))
    if any(any(receipt[key] != first[key] for key in ('source', 'acquisition', 'source_rows'))
           for receipt in receipts.values()):
        raise ValueError('static tokenizer views do not retain identical source acquisition obligations')
    parquet = pq.ParquetFile(first['source']['path'])
    if parquet.metadata.num_rows != first['source_rows']:
        raise ValueError('static source row obligation changed')
    def rows():
        position = 0
        for batch in parquet.iter_batches(batch_size=16):
            for value in batch.to_pylist():
                yield position, value
                position += 1
    groups, invalid = [], {}
    for position, value in rows():
        try:
            groups.append(source_row(value, source_sha256=first['source']['sha256'], position=position)['source_groups'])
        except ValueError as error:
            invalid[position] = str(error)
    closed, _ = close_group_sets(groups, {k: protection[k] for k in ('held', 'quality', 'pending')})
    del groups
    with acquire_output_lease(out):
        if (out/'audit.json').exists():
            raise ValueError('completed static admission is immutable')
        counts, conversions = {}, []
        row_proof = out/'row-proof.jsonl'
        with row_proof.open('x') as proof_writer:
            for model, receipt in receipts.items():
                source_tokenizer = AutoTokenizer.from_pretrained(str(tokenizers[model]), local_files_only=True, trust_remote_code=False)
                source_fp = tokenizer_fingerprint(source_tokenizer)
                expanded = copy.deepcopy(source_tokenizer); roles = provision(expanded)
                expanded_fp = tokenizer_fingerprint(expanded)
                if receipt['tokenizer'] != dict(source_fingerprint=source_fp,
                        expanded_fingerprint=expanded_fp, role_token_ids=list(roles)):
                    raise ValueError('static replay tokenizer differs from original complete conversion')
                proofs, pending = json_rows(receipt['row_proofs']['path']), json_rows(receipt['source_pending']['path'])
                streams = {split: json_rows(ref['path']) for split, ref in receipt['outputs'].items()}
                observed, seen = Counter(), set()
                for position, value in rows():
                    if position in invalid:
                        next_exact(pending, dict(source_row=position, reason=invalid[position], source_row_sha256=canonical_json_sha256_hex(value)))
                        observed['invalid_source_pending'] += 1
                        continue
                    row = source_row(value, source_sha256=first['source']['sha256'], position=position)
                    split = disposition(row['source_groups'], closed)
                    if split == 'pending':
                        next_exact(pending, dict(source_row=position, reason='protected-quality-or-pending', source_groups=row['source_groups']))
                        observed['protected_source_pending'] += 1
                        continue
                    identity = split, row['provenance']['code_sha256']
                    if identity in seen:
                        next_exact(proofs, dict(source_row=position, upstream_code_sha256=value['sha256'],
                            code_sha256=row['provenance']['code_sha256'], disposition='represented-identical-raw-code-body', split=split))
                        observed['represented_code_digest'] += 1
                        continue
                    seen.add(identity)
                    graph = continuation_graph(row, split, source_tokenizer, expanded, source_fp, expanded_fp)
                    graph['conversion']['static_repository_source'] = {**row['provenance'],
                        'parquet': receipt['source'], 'acquisition': receipt['acquisition'],
                        'original_row_sha256': canonical_json_sha256_hex(value),
                        'target_origin': 'original complete code text; static next-token continuation',
                        'execution_or_answer_success_asserted': False}
                    tokens = len(graph['nodes'][0]['token_ids']); digest = canonical_json_sha256_hex(graph)
                    next_exact(streams[split], graph)
                    next_exact(proofs, dict(source_row=position, id=graph['id'], split=split,
                        upstream_code_sha256=value['sha256'], code_sha256=row['provenance']['code_sha256'],
                        graph_sha256=digest, tokens=tokens, provenance=row['provenance']))
                    proof_writer.write(json.dumps(dict(model=model, id=graph['id'], split=split,
                        graph_sha256=digest, tokens=tokens), sort_keys=True)+'\n')
                    observed[split] += 1
                if (dict(observed) != receipt['counts'] or sum(observed.values()) != receipt['source_rows']
                        or any(next(stream, None) is not None for stream in [proofs, pending, *streams.values()])):
                    raise ValueError('static source fidelity replay lost or added complete rows')
                counts[model] = dict(observed)
                conversions.append(dict(model=model, outputs=[{**ref, 'split': split}
                    for split, ref in receipt['outputs'].items()]))
        for ref in evidence.values():
            check(ref)
        if current_protection(Path(protection_path))['proof'] != protection['proof']:
            raise ValueError('static protection changed during all-row replay')
        audit = dict(schema='natlang.static-repository-admission-audit/1', status='passed',
            all_row_source_render_verified=True, verifier=binding(__file__), evidence=list(evidence.values()),
            row_proof=binding(row_proof), row_proof_sha256=sha256_file_hex(row_proof),
            by_model=counts, conversions=conversions, source=first['source'],
            training_admission=False, execution_qualified=False, communication_foundation_qualification=False)
        write_json_durable(out/'audit.json', audit)
        verify_audit(out/'audit.json')
        write_json_durable(out/'source-render-samples.json', dict(passed=True,
            cohort_audit={'path':str(out/'audit.json'), 'sha256':sha256_file_hex(out/'audit.json')},
            all_rows=True, static_source_replay=True, execution_qualified=False))
        return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, action='append', required=True)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--protection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    items = [value.split('=', 1) for value in args.model]
    if any(len(item) != 2 for item in items) or len(dict(items)) != len(items):
        raise ValueError('declare distinct model tokenizer bindings')
    result = prepare(args.receipt, dict(items), args.protection, args.out)
    print(json.dumps(dict(passed=True, counts=result['by_model'], training_admission=False)))


if __name__ == '__main__':
    main()
