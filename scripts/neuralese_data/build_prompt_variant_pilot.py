#!/usr/bin/env python3
"""Prepare source-bound single-action prompt variants; admission is separate."""
import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer
from natlang_neuralese.common.jsonio import open_jsonl_text
from natlang_neuralese.data.native_graph import parse_graph
from natlang_neuralese.data.native_graph_conversion import text_row_graph
from natlang_neuralese.data.native_sources import _row_bindings
from natlang_neuralese.data.text_corpus import native_gold_packet, tokenizer_fingerprint
from natlang_neuralese.serve.chat import bind_history_reasoning
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import captured_text_emulation_context
from scripts.neuralese_data.prompt_variant_render import render_variant


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(request, audit, tokenizer_path, output):
    requested = json.loads(Path(request).read_text())
    audited = {r['id']: r for r in json.loads(Path(audit).read_text())['rows']}
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    bind_history_reasoning(tokenizer, policy='keep', require_declared=True, allow_transition=True)
    source_fp = tokenizer_fingerprint(tokenizer)
    provision(tokenizer)
    expanded_fp = tokenizer_fingerprint(tokenizer)
    result = []
    for selected in requested['rows']:
        item = audited[selected['id']]
        if item['binding'] != selected['binding'] or selected['split'] != 'train':
            raise ValueError('selected source binding or split differs')
        for evidence in selected['evidence']:
            if digest(evidence['path']) != evidence['sha256']:
                raise ValueError('selected evidence changed')
        teacher = item['teacher_source']
        if digest(item['source_file']) != selected['source_sha256']:
            raise ValueError('original graph source changed')
        oracle_paths = [e['path'] for e in selected['evidence'] if Path(e['path']).name == 'frontier-exact-oracle.json']
        if len(oracle_paths) != 1:
            raise ValueError('missing exact oracle evidence')
        oracle_rows = [r for r in json.loads(Path(oracle_paths[0]).read_text())['rows'] if r['id'] == selected['id']]
        if len(oracle_rows) != 1 or oracle_rows[0].get('exact_oracle_accepted') is not True:
            raise ValueError('selected action lacks exact accepted oracle')
        path = Path(teacher['source_artifact'])
        if digest(path) != teacher['source_artifact_sha256']:
            raise ValueError('trajectory source changed')
        with path.open('rb') as f:
            matches = [(raw.rstrip(b'\r\n'), json.loads(raw)) for raw in f if raw.strip()]
        matches = [(raw, row) for raw, row in matches if row['id'] == teacher['source_row_id']]
        if len(matches) != 1 or hashlib.sha256(matches[0][0]).hexdigest() != teacher['source_row_sha256']:
            raise ValueError('trajectory row identity differs')
        turn = matches[0][1]['trajectory'][teacher['teacher_turn_index']]
        target = {'role': 'assistant', 'content': turn['assistant'].get('content', ''),
                  'tool_calls': turn['model_response']['raw_calls']}
        if turn['assistant'].get('reasoning'):
            target['reasoning_content'] = turn['assistant']['reasoning']
        context, tools, _, omission = captured_text_emulation_context(turn, turn['context'], turn['tools_offered'])
        if omission is not None:
            raise ValueError('captured request cannot be faithfully expanded')
        with open_jsonl_text(item['source_file']) as stream:
            originals = [json.loads(line) for line in stream if selected['id'] in line]
        if len(originals) != 1:
            raise ValueError('expected exactly one admitted original graph')
        original = originals[0]
        parsed = parse_graph(original, tokenizer_fingerprint=expanded_fp)
        if _row_bindings([parsed])[selected['id']] != selected['binding']:
            raise ValueError('original graph binding changed')
        if len(original['nodes']) != 1:
            raise ValueError('source action is not a single invocation')
        originals_by_mode = [native_gold_packet(tokenizer, context, target, tools, prompt_rendering=mode)
                             for mode in ('program-header', 'full')]
        matched = [packet for packet in originals_by_mode if packet[1] == original['nodes'][0]['token_ids']]
        if not matched:
            raise ValueError('original historical rendering does not reproduce')
        expected_target = matched[0][1][matched[0][2]:]
        for mode in ('full', 'program-header'):
            text, ids, boundary = render_variant(tokenizer, context, target, tools, mode)
            if ids[boundary:] != expected_target:
                raise ValueError('variant changes original complete target')
            identifier = selected['id'] + ':tool-contract-prompt-variant:' + mode
            row = {'id': identifier, 'split': 'train', 'source_groups': original['source_groups'],
                   'tokenizer_sha256': source_fp, 'text': text, 'token_ids': ids,
                   'supervised_suffix_start': boundary, 'source_twin_id': identifier,
                   'source_record_ids': [teacher['source_row_id'], identifier]}
            graph = text_row_graph(row, source_tokenizer_fingerprint=source_fp, tokenizer=tokenizer,
                                   expanded_tokenizer_fingerprint=expanded_fp, context_weight=0.)
            graph['conversion']['prompt_variant'] = {'schema': 'natlang.prompt-variant-source/1',
                'original_id': selected['id'], 'original_binding': selected['binding'],
                'original_source_sha256': selected['source_sha256'], 'mode': mode,
                'offered_tools_sha256': hashlib.sha256(json.dumps(tools, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest(),
                'teacher_source': teacher}
            result.append(graph)
    output = Path(output)
    with output.open('x') as f:
        for row in result:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    return len(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for option in ('request', 'audit', 'tokenizer', 'output'):
        p.add_argument('--' + option, required=True)
    args = p.parse_args()
    print(json.dumps({'rows': build(args.request, args.audit, args.tokenizer, args.output), 'admitted': False}))
