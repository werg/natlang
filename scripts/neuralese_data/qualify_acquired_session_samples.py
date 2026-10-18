"""Source-backed exact re-render spotchecks over an audited complete cohort."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.build_acquired_full_session_graphs import sessions, session_graphs, prepare_tokenizer, numerical_runtime
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


def proposal_with_samples(audit_path, sample_path):
    """Prepare a final review-only proposal; never modify the live registry."""
    audit_path, sample_path = Path(audit_path), Path(sample_path)
    proposal_path = audit_path.parent / 'registry-proposal.json'
    proposal = json.loads(proposal_path.read_text()); root = audit_path.parent.parent
    sample = json.loads(sample_path.read_text())
    audit_binding = {'path': str(audit_path), 'sha256': sha256_file_hex(audit_path)}
    if (sample.get('passed') is not True or sample.get('cohort_audit') != audit_binding
            or proposal['build']['audit_sha256'] != audit_binding['sha256']):
        raise ValueError('source sample proof differs from its reviewed cohort proposal')
    relative = str(sample_path.resolve().relative_to(root.resolve()))
    proposal['include'] = sorted(set(proposal['include']) | {relative})
    proposal['build']['source_render_samples'] = {'path': relative, 'sha256': sha256_file_hex(sample_path)}
    target = audit_path.parent / 'registry-proposal-with-render-samples.json'
    write_json_durable(target, proposal)
    return target


@contextmanager
def conversion_threads(recorded, override=None):
    """Use recorded per-shard threads; legacy settings remain explicitly unknown."""
    import torch
    previous=torch.get_num_threads()
    target=override if override is not None else (recorded or {}).get('torch_threads')
    if target is not None and (type(target) is not int or target<1):
        raise ValueError('source reference torch threads must be a positive integer')
    try:
        if target is not None and target!=previous:torch.set_num_threads(target)
        yield numerical_runtime()
    finally:
        if torch.get_num_threads()!=previous:torch.set_num_threads(previous)


def recorded_runtime(receipt, ordinal):
    for shard in receipt.get('shards') or []:
        if shard['start']<=ordinal<shard['stop']:
            path=bound(shard['path'],shard)
            return (json.loads(path.read_text()).get('runtime') or {}).get('numerical_runtime')
    return (receipt.get('runtime') or {}).get('numerical_runtime')


def declared_history_profile(conversion_receipts):
    """Return the one profile authenticated by every model conversion receipt."""
    profiles = {receipt.get('explicit_history_reasoning_policy')
                for receipt in conversion_receipts}
    if len(profiles) != 1:
        raise ValueError('source sample cohort mixes incompatible history-rendering profiles')
    profile = next(iter(profiles))
    if profile not in (None, 'keep', 'last_turn_only'):
        raise ValueError('source sample cohort declares an unsupported history-rendering profile')
    return profile


def qualify(audit_path, models, out, *, torch_threads=None):
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
        model_conversions = [item for item in audit['conversions'] if item['model'] == model]
        authenticated_receipts = [json.loads(bound(item['receipt']['path'], item['receipt']).read_text())
                                  for item in model_conversions]
        profile = declared_history_profile(authenticated_receipts)
        prepared = prepare_tokenizer(base, history_reasoning_policy=profile)
        picks = {proof['normalized_session_sha256']: proof for proof in selected[model].values()}
        picked_original_files = {proof['original_source']['sha256'] for proof in picks.values()}
        observed = {key: [] for key in picks}; original = {}; receipts = []; numerical = {}
        for conversion in model_conversions:
            receipt_path = bound(conversion['receipt']['path'], conversion['receipt'])
            receipt = json.loads(receipt_path.read_text())
            if receipt.get('explicit_history_reasoning_policy') != profile:
                raise ValueError('source sample conversion profile changed after authentication')
            if receipt['prepared_tokenizer'] != prepared.binding:
                raise ValueError('source-backed sample uses a different actual tokenizer/rendering configuration')
            export_path = bound(receipt['export_receipt']['path'], receipt['export_receipt'])
            if json.loads(export_path.read_text())['contract']['source']['sha256'] not in picked_original_files:
                continue
            source_path = bound(receipt['source']['path'], receipt['source'])
            for ordinal,value in enumerate(sessions(source_path)):
                digest = canonical_json_sha256_hex(value)
                if digest in picks:
                    if digest in original: raise ValueError('sample source occurrence is duplicated')
                    original[digest] = value
                    numerical[digest] = recorded_runtime(receipt,ordinal)
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
            with conversion_threads(numerical[digest],torch_threads) as reference_runtime:
                expected = list(session_graphs(value, prepared.tokenizer,
                    source_fingerprint=prepared.source_fingerprint,
                    expanded_fingerprint=prepared.expanded_fingerprint, split=proof['split'],
                    explicit_history_reasoning_policy=profile))
            views = compare_views(expected, observed[digest])
            reports.append({'model': model, 'selection': [kind for kind, selected_proof in selected[model].items()
                if selected_proof['normalized_session_sha256'] == digest],
                'normalized_session_sha256': digest, 'original_source': value['source_file'],
                'original_source_row': value['source_row'], 'source_outcome': value['trajectory']['outcome'],
                'tokenizer': prepared.binding,
                'declared_history_reasoning_profile': profile or 'backbone-default',
                'bindings': [r for r in receipts if r['normalized_session_sha256'] == digest],
                'exact_entire_graph_equal': True, 'views': views,
                'recorded_conversion_numerical_runtime': numerical[digest],
                'reference_numerical_runtime': reference_runtime,
                'explicit_reference_thread_override': torch_threads})
        prepared.validate(base)
    if sha256_file_hex(audit_path) != before: raise ValueError('cohort audit changed during source-backed samples')
    result = {'schema': 'natlang.acquired-session-source-render-samples/2', 'passed': True,
        'cohort_audit': {'path': str(audit_path), 'sha256': before}, 'samples': reports,
        'scope': 'first-positive and largest whole-session positive per model; exact full graph/view rerender under each authenticated profile; '
                 'not a full-cohort rerender, task oracle, native execution or student exposure qualification',
        'verifier_sha256': sha256_file_hex(__file__)}
    write_json_durable(out, result)
    proposal = proposal_with_samples(audit_path, out)
    print(json.dumps({'passed': True, 'samples': len(reports), 'path': str(out),
                     'sha256': sha256_file_hex(out), 'registry_proposal': str(proposal), 'registry_written': False}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True, help='NAME=actual pinned tokenizer base')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--torch-threads', type=int,
        help='explicit current reference for legacy conversions whose original setting was not recorded')
    args = parser.parse_args(argv)
    items = [item.split('=', 1) for item in args.model]
    if any(len(item) != 2 for item in items) or len(dict(items)) != len(items):
        raise ValueError('models require distinct NAME=actual pinned tokenizer base bindings')
    qualify(args.audit, dict(items), args.out, torch_threads=args.torch_threads)


if __name__ == '__main__':
    main()
