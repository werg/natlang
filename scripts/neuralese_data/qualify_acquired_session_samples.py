"""Source-backed exact re-render spotchecks over an audited complete cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.build_acquired_full_session_graphs import sessions, session_graphs, prepare_tokenizer
from scripts.neuralese_data.prepare_acquired_session_admission import bound


def compare_views(expected, stored):
    """Compare every derived view and the entire original token/weight graph."""
    actual = {graph['id']: graph for graph in stored}
    reference = {graph['id']: graph for graph in expected}
    if len(actual) != len(stored) or len(reference) != len(expected) or set(actual) != set(reference):
        raise ValueError('source-backed sample view inventory differs from stored native graphs')
    for identifier in sorted(reference):
        if canonical_json_sha256_hex(actual[identifier]) != canonical_json_sha256_hex(reference[identifier]):
            raise ValueError('source-backed sample graph tokens/weights/provenance differ: ' + identifier)
    return [{'id': identifier, 'sha256': canonical_json_sha256_hex(actual[identifier]),
        'tokens': sum(len(node['token_ids']) for node in actual[identifier]['nodes'])}
        for identifier in sorted(actual)]


def qualify(audit_path, models, out):
    audit_path = Path(audit_path); before = sha256_file_hex(audit_path)
    audit = json.loads(audit_path.read_text())
    if (audit.get('schema') != 'natlang.acquired-session-admission-audit/1'
            or audit.get('status') != 'passed' or set(models) != set(audit['by_model'])):
        raise ValueError('sample qualification requires the actual complete cohort audit and both model views')
    proof_path = bound(audit_path.parent / 'row-proof.jsonl', {'sha256': audit['row_proof_sha256']})
    selected = {}
    for proof in sessions(proof_path):
        if proof['view'] != 'whole-trajectory': continue
        model = proof['model']; value = selected.setdefault(model, {'first-positive': proof, 'largest-positive': proof})
        if proof['tokens'] > value['largest-positive']['tokens']: value['largest-positive'] = proof
    if set(selected) != set(models): raise ValueError('both models need actual admitted positive source samples')
    reports = []
    for model, base in models.items():
        prepared = prepare_tokenizer(base)
        picks = {proof['normalized_session_sha256']: proof for proof in selected[model].values()}
        picked_original_files = {proof['original_source']['sha256'] for proof in picks.values()}
        observed = {key: [] for key in picks}; original = {}; receipts = []
        for conversion in audit['conversions']:
            if conversion['model'] != model: continue
            receipt_path = bound(conversion['receipt']['path'], conversion['receipt'])
            receipt = json.loads(receipt_path.read_text())
            if receipt['prepared_tokenizer'] != prepared.binding:
                raise ValueError('source-backed sample uses a different actual tokenizer/rendering configuration')
            export_path = bound(receipt['export_receipt']['path'], receipt['export_receipt'])
            if json.loads(export_path.read_text())['contract']['source']['sha256'] not in picked_original_files:
                continue
            source_path = bound(receipt['source']['path'], receipt['source'])
            for value in sessions(source_path):
                digest = canonical_json_sha256_hex(value)
                if digest in picks:
                    if digest in original: raise ValueError('sample source occurrence is duplicated')
                    original[digest] = value
                    receipts.append({'normalized_session_sha256': digest, 'source': receipt['source'],
                        'conversion_receipt': conversion['receipt']})
            for output in conversion['outputs']:
                path = bound(output['path'], output)
                for graph in sessions(path):
                    digest = graph['conversion']['acquired_session']['normalized_session_sha256']
                    if digest in picks: observed[digest].append(graph)
        if set(original) != set(picks): raise ValueError('sample lost its exact original normalized source')
        for digest, proof in picks.items():
            value = original[digest]
            expected = list(session_graphs(value, prepared.tokenizer,
                source_fingerprint=prepared.source_fingerprint,
                expanded_fingerprint=prepared.expanded_fingerprint, split=proof['split']))
            views = compare_views(expected, observed[digest])
            reports.append({'model': model, 'selection': [kind for kind, selected_proof in selected[model].items()
                if selected_proof['normalized_session_sha256'] == digest],
                'normalized_session_sha256': digest, 'original_source': value['source_file'],
                'original_source_row': value['source_row'], 'source_outcome': value['trajectory']['outcome'],
                'tokenizer': prepared.binding, 'bindings': [r for r in receipts if r['normalized_session_sha256'] == digest],
                'exact_entire_graph_equal': True, 'views': views})
        prepared.validate(base)
    if sha256_file_hex(audit_path) != before: raise ValueError('cohort audit changed during source-backed samples')
    result = {'schema': 'natlang.acquired-session-source-render-samples/1', 'passed': True,
        'cohort_audit': {'path': str(audit_path), 'sha256': before}, 'samples': reports,
        'scope': 'first-positive and largest whole-session positive per model; exact full graph/view rerender; '
                 'not a full-cohort rerender, task oracle, native execution or student exposure qualification',
        'verifier_sha256': sha256_file_hex(__file__)}
    write_json_durable(out, result)
    print(json.dumps({'passed': True, 'samples': len(reports), 'path': str(out), 'sha256': sha256_file_hex(out)}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True, help='NAME=actual pinned tokenizer base')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    items = [item.split('=', 1) for item in args.model]
    if any(len(item) != 2 for item in items) or len(dict(items)) != len(items):
        raise ValueError('models require distinct NAME=actual pinned tokenizer base bindings')
    qualify(args.audit, dict(items), args.out)


if __name__ == '__main__':
    main()
