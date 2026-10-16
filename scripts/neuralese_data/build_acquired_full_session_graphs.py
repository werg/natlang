"""Complete source-curated sessions through existing serving/native renderers."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.data.native_graph_conversion import text_row_graph, whole_trajectory_text_row_graph
from natlang_neuralese.data.text_corpus import (native_gold_document, native_gold_packet,
    native_gold_rendered_text, tokenizer_fingerprint)
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds


def sessions(path):
    with open_jsonl_text(path) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def source_target_ids(value):
    """Original trajectory.record identity, before any derived graph ID."""
    trajectory = value['trajectory']
    return [trajectory['id'] + ':' + str(index) for index, message in enumerate(trajectory['messages'])
            if message.get('role') == 'assistant']


def held_session_targets(value, held_sources):
    trajectory = value['trajectory']
    identities = {trajectory['id'], *source_target_ids(value)}
    return sorted(identities & held_sources)


def session_graphs(value, tokenizer, *, source_fingerprint, expanded_fingerprint, split):
    """Keep the full session and add packets only for reasoning hidden by serving."""
    if (value.get('schema') != 'natlang.normalized-agent-trajectory/1'
            or value.get('whole_trajectory') is not True or value.get('windowed') is not False):
        raise ValueError('expected a complete normalized source session')
    trajectory = value['trajectory']
    if value.get('positive_candidate') is not True or trajectory['outcome'].get('label') == 'failed':
        raise ValueError('non-positive source session is not positive SFT')
    from natlang_neuralese.serve.chat import bind_history_reasoning
    bind_history_reasoning(tokenizer)
    history_policy = tokenizer.natlang_history_reasoning['policy']
    messages = []
    for original in trajectory['messages']:
        item = dict(original)
        if 'reasoning' in item:
            item['reasoning_content'] = item.pop('reasoning')
        messages.append(item)
    assistants = [i for i, item in enumerate(messages) if item['role'] == 'assistant']
    if not assistants:
        raise ValueError('source session has no assistant targets')
    occurrence = canonical_json_sha256_hex([value['source_file']['sha256'], value['source_row']])
    identifier = trajectory['id'] + ':source-occurrence:' + occurrence
    common = {'split': split, 'source_groups': trajectory['split_groups'],
              'tokenizer_sha256': source_fingerprint, 'source_record_ids': [trajectory['id']]}
    convert = {'source_tokenizer_fingerprint': source_fingerprint, 'tokenizer': tokenizer,
               'expanded_tokenizer_fingerprint': expanded_fingerprint}
    text, ids = native_gold_document(tokenizer, messages[:-1], messages[-1], trajectory.get('tools'))
    whole = {**common, 'id': identifier, 'supervision': 'whole-trajectory', 'text': text, 'token_ids': ids}
    graph = whole_trajectory_text_row_graph(whole, **convert)
    # Probe the actual full-conversation template one message at a time. A unique
    # sentinel travels in the original reasoning field, so duplicate text in a
    # user/system/tool message cannot make an omitted assistant turn look covered.
    # This render-only pass uses the same serving path as native_gold_document.
    probe_messages = [dict(message) for message in messages]
    sentinels = {}
    original_render = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    for ordinal, index in enumerate(assistants):
        reasoning = messages[index].get('reasoning_content')
        if not isinstance(reasoning, str) or not reasoning:
            continue
        sentinel = 'NATLANG_REASONING_COVERAGE_' + canonical_json_sha256_hex(
            [occurrence, index, reasoning]) + '_END'
        if sentinel in original_render or sentinel in sentinels.values():
            raise ValueError('reasoning coverage sentinel collides with source content')
        probe_messages[index]['reasoning_content'] = sentinel
        sentinels[ordinal] = sentinel
    probed_text = native_gold_rendered_text(tokenizer, probe_messages, trajectory.get('tools'))
    reasoning_coverage = {str(ordinal): {'present_in_whole_view': probed_text.count(sentinel) == 1,
        'sentinel_occurrences': probed_text.count(sentinel)} for ordinal, sentinel in sentinels.items()}
    if any(item['sentinel_occurrences'] > 1 for item in reasoning_coverage.values()):
        raise ValueError('a source reasoning sentinel rendered more than once')
    auxiliary_reasoning_ordinals = [ordinal for ordinal, item in reasoning_coverage.items()
        if not item['present_in_whole_view']]
    auxiliary_reasoning_ordinals = [int(ordinal) for ordinal in auxiliary_reasoning_ordinals]
    provenance = {'normalized_session_sha256': canonical_json_sha256_hex(value),
        'original_source_row': value['source_row'], 'original_target_ids': source_target_ids(value),
        'original_source_file': value['source_file'], 'source_revision': value['source_revision'],
        'source_lineage': trajectory['lineage'], 'source_license': trajectory['license'],
        'original_messages': len(messages), 'original_assistant_turns': len(assistants),
        'admission_class': 'source-curated-third-party-SFT', 'oracle_verified': False,
        'native_execution_verified': False, 'observed_actor_state_exchange': False,
        'history_reasoning': tokenizer.natlang_history_reasoning,
        'reasoning_coverage_probe': 'per-assistant-unique-sentinel/full-serving-render/1',
        'reasoning_coverage': reasoning_coverage,
        'whole_view_reasoning_assistant_ordinals': [ordinal for ordinal, index in enumerate(assistants)
            if messages[index].get('reasoning_content') and ordinal not in auxiliary_reasoning_ordinals],
        'auxiliary_reasoning_assistant_ordinals': auxiliary_reasoning_ordinals,
        'truncated': False}
    graph['conversion']['acquired_session'] = {**provenance, 'view': 'whole-trajectory'}
    yield graph
    for ordinal, index in enumerate(assistants):
        if ordinal not in auxiliary_reasoning_ordinals:
            continue
        text, ids, boundary = native_gold_packet(tokenizer, messages[:index], messages[index], trajectory.get('tools'))
        row = {**common, 'id': identifier + ':reasoning-turn:' + str(ordinal),
               'text': text, 'token_ids': ids, 'supervised_suffix_start': boundary}
        graph = text_row_graph(row, **convert)
        graph['conversion']['acquired_session'] = {**provenance,
            'view': 'full-prefix-reasoning-target', 'assistant_turn': ordinal,
            'whole_view_reasoning_coverage': reasoning_coverage[str(ordinal)],
            'history_reasoning_policy': history_policy}
        yield graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-receipt', type=Path, required=True)
    parser.add_argument('--base-tokenizer', type=Path, required=True)
    parser.add_argument('--protection', type=Path, required=True)
    parser.add_argument('--decision-hold', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export = json.loads(args.export_receipt.read_text())
    if export.get('schema') != 'natlang.normalized-agent-source-file/1':
        raise ValueError('source needs its complete normalized export receipt')
    source = args.export_receipt.parent / 'full-trajectories.jsonl.zst'
    digest = sha256_file_hex(source)
    if digest != export['outputs'][source.name]:
        raise ValueError('complete session export differs from its receipt')
    protection = current_protection(args.protection)
    current_holds = sorted((Path(__file__).resolve().parents[2] / 'training' / 'decision-holds').glob('*.jsonl'))
    if {path.resolve() for path in args.decision_hold} != {path.resolve() for path in current_holds}:
        raise ValueError('decision hold bindings must include the complete current file set')
    decision_ids, decision_bindings = load_decision_holds(current_holds)
    held_sources = {identifier.split(':decision:', 1)[0] for identifier in decision_ids} | decision_ids
    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, local_files_only=True, trust_remote_code=False)
    source_fp = tokenizer_fingerprint(tokenizer)
    provision(tokenizer)
    expanded_fp = tokenizer_fingerprint(tokenizer)
    # Complete source-file held-group closure before writing either split. Final
    # corpus admission also closes groups across all source files and views.
    held = set(protection['held'])
    for value in sessions(source):
        trajectory = value['trajectory']
        if trajectory['split'] not in ('train', 'test', 'validation'):
            raise ValueError('unknown source session split must not become train')
        if (value['source_file'] != export['contract']['source']
                or value['source_revision'] != export['contract']['revision']):
            raise ValueError('normalized session differs from original export source binding')
        if trajectory['split'] in ('test', 'validation'):
            held.update(trajectory['split_groups'])
    if (args.output / 'receipt.json').exists():
        raise ValueError('completed graph output is immutable; use its bound receipt')
    writers = {split: GraphWriter(args.output / split / 'native-graphs.jsonl.zst') for split in ('train', 'test')}
    counts = Counter()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        with (args.output / 'omissions.jsonl').open('w') as omissions:
            for value in sessions(source):
                trajectory = value['trajectory']; identifier = trajectory['id']; groups = set(trajectory['split_groups'])
                reason = None
                if not value.get('positive_candidate') or trajectory['outcome'].get('label') == 'failed':
                    reason = 'source-reported-non-positive'
                elif groups & (protection['quality'] | protection['pending']):
                    reason = 'current-source-quality-or-pending-protection'
                elif held_session_targets(value, held_sources):
                    reason = 'current-decision-hold'
                elif not source_target_ids(value):
                    reason = 'source-session-without-assistant-target'
                if reason:
                    omissions.write(json.dumps({'id': identifier, 'source_row': value['source_row'], 'reason': reason,
                        'original_target_ids': source_target_ids(value),
                        'matched_hold_sources': held_session_targets(value, held_sources),
                        'hold_matching_scope': 'conservative original source/target identity prefix; no claim of raw-row hash equality',
                        'sibling_targets_retained_in_bound_source': reason == 'current-decision-hold'})+'\n')
                    counts[reason] += 1
                    continue
                split = 'test' if groups & held else 'train'
                rendered = session_graphs(value, tokenizer, source_fingerprint=source_fp,
                        expanded_fingerprint=expanded_fp, split=split)
                for graph in rendered:
                    writers[split].write(graph)
                    counts['graphs:'+split] += 1
                    if (graph.get('conversion', {}).get('acquired_session', {}).get('view')
                            == 'full-prefix-reasoning-target'):
                        counts['auxiliary_reasoning_graphs:'+split] += 1
                counts['positive_sessions'] += 1
        outputs = [writer.finish() for writer in writers.values()]
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    if sum(counts[key] for key in ('positive_sessions', 'source-reported-non-positive',
            'current-source-quality-or-pending-protection', 'current-decision-hold',
            'source-session-without-assistant-target')) != export['complete_sessions']:
        raise ValueError('complete normalized source session accounting differs')
    if sha256_file_hex(source) != digest:
        raise ValueError('source changed during rendering')
    current_protection(args.protection)
    if (sorted((Path(__file__).resolve().parents[2] / 'training' / 'decision-holds').glob('*.jsonl')) != current_holds
            or load_decision_holds(current_holds)[1] != decision_bindings):
        raise ValueError('current decision holds changed during rendering')
    write_json_durable(args.output / 'receipt.json', {'schema': 'natlang.acquired-full-session-graphs/1',
        'source': {'path': str(source), 'sha256': digest},
        'export_receipt': {'path': str(args.export_receipt), 'sha256': sha256_file_hex(args.export_receipt)},
        'protection': protection['proof'], 'decision_holds': decision_bindings,
        'source_tokenizer_fingerprint': source_fp, 'expanded_tokenizer_fingerprint': expanded_fp,
        'builder_sha256': sha256_file_hex(__file__),
        'counts': dict(counts), 'outputs': outputs, 'training_admission': False,
        'omissions_sha256': sha256_file_hex(args.output / 'omissions.jsonl')})


if __name__ == '__main__':
    main()
