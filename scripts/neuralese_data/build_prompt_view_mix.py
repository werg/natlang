#!/usr/bin/env python3
"""Prompt-view mixture of serving-prompt renders, and its trainer-load verification (CPU).

``mix``: from one render's native graphs (render_native_turn_corpus.py --serving-entry, converted by
build_native_graph_text_corpus.py at context weight 0), write the TRAIN mixture: each decision document appears once,
in its program-header view when sha256('prompt-view', id) falls below ``--header-share``, else in its full serving
view (whose entry point, collector or runtime, the render already drew per record); every whole-trajectory document
is kept in its serving view. Held (test) rows are never read or written here. Shares, counts and the exact inputs go
to OUT/receipt.json. Selection never changes row bytes, ids, splits or source bindings.

``verify``: load native graph files the way the trainer does (``NativeGraphSource`` parse/validation, the packed
geometry index ``prepare_source``) and apply the default ``own-outputs/1`` objective to every row: no position before
the first assistant decision may carry raw or objective weight, every row must keep a target. Reports rows, tokens,
prompt tokens (before the first assistant turn), the token count of the prefix trie of all rows and of their prompts
(every shared causal prefix counted once) and the prompt-view distribution.

  build_prompt_view_mix.py mix --render RENDER_DIR --out OUT [--header-share 0.2]
  build_prompt_view_mix.py verify --source F [F ...] --base-tokenizer DIR --out REPORT.json [--geometry-dir D]
      [--ids-from F ...] [--compare-with F ...] [--report-only]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))

HEADER_SHARE = 0.2


def header_view(row_id: str, share: float) -> bool:
    draw = int.from_bytes(hashlib.sha256(('prompt-view\0' + row_id).encode()).digest()[:8], 'big') / 2 ** 64
    return draw < share


def trie_tokens(sequences) -> int:
    """Tokens of the prefix trie of ``sequences``: what remains when every shared causal prefix is counted once."""
    total, previous = 0, ()
    for sequence in sorted(sequences):
        common = 0
        for a, b in zip(sequence, previous):
            if a != b:
                break
            common += 1
        total += len(sequence) - common
        previous = sequence
    return total


def rows(path):
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    for _, row in iter_jsonl_rows(path):
        yield row


def mix(args):
    from build_native_graph_text_corpus import GraphWriter
    from natlang_neuralese.common.hashing import sha256_file_hex
    graphs = args.render / 'native-graphs'
    inputs = {view: graphs / view / 'train' / 'native-graphs.jsonl.zst'
              for view in ('full', 'program-header', 'whole-trajectory')}
    present = {view: path for view, path in inputs.items() if path.exists()}
    header = {row['id']: row for row in rows(present['program-header'])} if 'program-header' in present else {}
    counts, modes = Counter(), Counter()
    writer = GraphWriter(args.out / 'train' / 'native-graphs.jsonl.zst')
    try:
        seen = set()
        for row in rows(present['full']) if 'full' in present else ():
            choose_header = row['id'] in header and header_view(row['id'], args.header_share)
            chosen = header[row['id']] if choose_header else row
            counts['decision:' + ('program-header' if choose_header else 'full')] += 1
            modes[(chosen['conversion'].get('prompt_view') or {}).get('system_prompt_mode', 'none')] += 1
            seen.add(row['id'])
            writer.write(chosen)
        for row_id, row in header.items():
            if row_id not in seen:  # a decision whose full view the renderer omitted keeps its header view
                counts['decision:program-header-only'] += 1
                writer.write(row)
        for row in rows(present['whole-trajectory']) if 'whole-trajectory' in present else ():
            counts['whole-trajectory'] += 1
            modes[(row['conversion'].get('prompt_view') or {}).get('system_prompt_mode', 'none')] += 1
            writer.write(row)
        summary = writer.finish()
    except BaseException:
        writer.abort()
        raise
    receipt = {'schema': 'natlang.prompt-view-mix/1', 'render': str(args.render), 'header_share': args.header_share,
               'selection': "decision rows: program-header iff sha256('prompt-view\\0'+id) < header_share, else "
                            'the full serving view; whole-trajectory rows: serving view; held rows untouched',
               'inputs': {view: {'path': str(path), 'sha256': sha256_file_hex(path)} for view, path in present.items()},
               'counts': dict(counts), 'system_prompt_modes': dict(modes), 'output': summary,
               'training_admission_granted': False}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({key: receipt[key] for key in ('counts', 'system_prompt_modes')}))


def expanded_tokenizer(path):
    from transformers import AutoTokenizer
    from natlang_neuralese.model.shared_state import ROLE_MARKERS
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=False, local_files_only=True)
    import inspect
    modern = 'replace_extra_special_tokens' in inspect.signature(tokenizer.add_special_tokens).parameters
    tokenizer.add_special_tokens({('extra_special_tokens' if modern else 'additional_special_tokens'): list(ROLE_MARKERS)},
                                 **{('replace_extra_special_tokens' if modern else 'replace_additional_special_tokens'): False})
    roles = [tokenizer.convert_tokens_to_ids(marker) for marker in ROLE_MARKERS]
    return tokenizer, tokenizer_fingerprint(tokenizer), roles


def verify(args):
    from array import array
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.data.native_sources import NativeGraphSource
    from natlang_neuralese.train.native_packed_capacity import prepare_source
    from natlang_neuralese.train.own_output_supervision import default_objective, graph_weights, label_roles
    tokenizer, fingerprint, roles = expanded_tokenizer(args.base_tokenizer)
    objective, reason = default_objective(tokenizer, tokenizer_fingerprint=fingerprint)
    if objective is None:
        raise ValueError(f'no own-output objective: {reason}')
    assistant = objective['role_tokens']['roles']['assistant']
    start = objective['role_tokens']['start']
    report = {'schema': 'natlang.prompt-view-graph-verification/1',
              'verifier_sha256': sha256_file_hex(Path(__file__)), 'tokenizer_fingerprint': fingerprint,
              'objective': objective['name'], 'files': []}
    failures = 0
    wanted = None
    if args.ids_from:
        wanted = {row['id'] for path in args.ids_from for row in rows(path)}
        report['ids_from'] = [str(path) for path in args.ids_from]
    user = objective['role_tokens']['roles']['user']

    def after_system(ids):
        """Tokens from the first user turn on: everything a prompt view leaves unchanged."""
        at = next((i for i in range(len(ids) - 1) if ids[i] == start and ids[i + 1] == user), None)
        return None if at is None else ids[at:]
    compared = {}
    for path in args.compare_with or ():
        for row in rows(path):
            compared[row['id']] = row
    for path in args.source:
        source = NativeGraphSource(path, fingerprint, required_split='train', role_ids=roles)
        geometry = None
        if args.geometry_dir:
            out = args.geometry_dir / hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
            out.mkdir(parents=True, exist_ok=True)
            _, _, geometry_reference, scan = prepare_source(path, fingerprint, roles, out)
            geometry = {'reference': geometry_reference, 'scan': {k: v for k, v in scan.items() if not isinstance(v, (list, dict))}}
        stats, modes, problems = Counter(), Counter(), []
        sequences, prompts = [], []
        for row in rows(path):
            if wanted is not None and row['id'] not in wanted:
                continue
            ids = row['nodes'][0]['token_ids']
            first = next((i for i in range(len(ids) - 1) if ids[i] == start and ids[i + 1] == assistant), len(ids))
            raw = [w for node in row['nodes'] for w in node['target_weights']]
            weights, _ = graph_weights(row, objective)
            stats['rows'] += 1
            stats['tokens'] += sum(len(node['token_ids']) for node in row['nodes'])
            stats['prompt_tokens'] += first
            # Keep exact lexicographic prefix counting without retaining Python
            # integer objects for every token in an entire cohort.
            sequences.extend(array('I', node['token_ids']) for node in row['nodes'])
            prompts.append(array('I', ids[:first]))
            stats['objective_target_tokens'] += sum(1 for w in weights if w > 0)
            view = (row['conversion'].get('prompt_view') or {})
            modes[f"{view.get('prompt_rendering', 'undeclared')}/{view.get('system_prompt_mode', 'none')}"] += 1
            if any(w > 0 for w in raw[:first]) or any(w > 0 for w in weights[:first]):
                problems.append({'id': row['id'], 'problem': 'weight before the first assistant turn'})
            if not any(w > 0 for w in weights):
                problems.append({'id': row['id'], 'problem': 'no own-output target'})
            other = compared.get(row['id'])
            if other is not None and len(other['nodes']) == len(row['nodes']) == 1:
                stats['compared'] += 1
                mine, theirs = after_system(ids), after_system(other['nodes'][0]['token_ids'])
                other_weights, _ = graph_weights(other, objective)
                if mine is None or mine != theirs:
                    problems.append({'id': row['id'], 'problem': 'tokens after the system turn differ from the '
                                                                 'compared rendering'})
                elif sum(w > 0 for w in weights) != sum(w > 0 for w in other_weights):
                    problems.append({'id': row['id'], 'problem': 'own-output target count differs from the '
                                                                 'compared rendering'})
                else:
                    stats['compared_equal'] += 1
        stats['unique_prefix_tokens'] = trie_tokens(sequences)
        stats['unique_prompt_prefix_tokens'] = trie_tokens(prompts)
        failures += len(problems)
        report['files'].append({'path': str(path), 'source_rows': len(source), **stats,
                                'prompt_views': dict(modes), 'geometry': geometry, 'problems': problems[:20],
                                'problem_count': len(problems)})
    report['passed'] = failures == 0
    report['report_only'] = args.report_only
    args.out.write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'passed': report['passed'], 'files': [{k: f.get(k) for k in (
        'path', 'rows', 'tokens', 'prompt_tokens', 'unique_prefix_tokens', 'unique_prompt_prefix_tokens',
        'prompt_views', 'compared', 'compared_equal', 'problem_count')}
        for f in report['files']]}, indent=1))
    sys.exit(0 if report['passed'] or args.report_only else 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    m = sub.add_parser('mix')
    m.add_argument('--render', type=Path, required=True)
    m.add_argument('--out', type=Path, required=True)
    m.add_argument('--header-share', type=float, default=HEADER_SHARE)
    v = sub.add_parser('verify')
    v.add_argument('--source', type=Path, nargs='+', required=True)
    v.add_argument('--base-tokenizer', type=Path, required=True)
    v.add_argument('--geometry-dir', type=Path)
    v.add_argument('--out', type=Path, required=True)
    v.add_argument('--ids-from', type=Path, nargs='*', help='only rows whose id occurs in these graph files')
    v.add_argument('--compare-with', type=Path, nargs='*',
                   help='graph files of another rendering of the same documents (e.g. the recorded-prompt or '
                        'program-header render): rows with the same id must have identical tokens from the first user '
                        'turn on and the same number of own-output targets')
    v.add_argument('--report-only', action='store_true',
                   help='measure without failing (e.g. older renders with nonzero raw context weights)')
    args = parser.parse_args()
    (mix if args.command == 'mix' else verify)(args)


if __name__ == '__main__':
    main()
