"""Complete source-curated sessions through existing serving/native renderers."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

from transformers import AutoTokenizer
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from natlang_neuralese.data.native_graph_conversion import text_row_graph, whole_trajectory_text_row_graph
from natlang_neuralese.data.text_corpus import (native_gold_document, native_gold_packet,
    native_gold_rendered_text, tokenizer_fingerprint)
from natlang_neuralese.model.native_model import tokenizer_inventory
from natlang_neuralese.serve.chat import bind_history_reasoning
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease


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


def session_exclusion(value, protection, held_sources):
    """Exact declared source eligibility shared by conversion and admission."""
    trajectory = value['trajectory']
    if not value.get('positive_candidate') or trajectory['outcome'].get('label') == 'failed':
        return 'source-reported-non-positive'
    if set(trajectory['split_groups']) & (protection['quality'] | protection['pending']):
        return 'current-source-quality-or-pending-protection'
    if held_session_targets(value, held_sources):
        return 'current-decision-hold'
    if not source_target_ids(value):
        return 'source-session-without-assistant-target'
    return None


def runtime_bindings(root=None):
    root = Path(root or Path(__file__).resolve().parents[2])
    package = root / 'training' / 'neuralese' / 'natlang_neuralese'
    paths = {
        'builder': Path(__file__).resolve(),
        'text_corpus': package / 'data' / 'text_corpus.py',
        'native_graph_conversion': package / 'data' / 'native_graph_conversion.py',
        'prompt_rendering': package / 'data' / 'prompt_rendering.py',
        'serve_chat': package / 'serve' / 'chat.py',
        'jsonio': package / 'common' / 'jsonio.py',
        'graph_writer': root / 'scripts' / 'neuralese_data' / 'build_native_graph_text_corpus.py',
        'tokenizer_provision': root / 'scripts' / 'neuralese_data' / 'build_native_semantic_pairs_e2e.py',
        'tokenizer_inventory': package / 'model' / 'native_model.py',
    }
    # Identity follows the logical runtime files and their bytes, not checkout
    # location. Keep observed paths as receipt metadata outside the fingerprint.
    files = {name: {'sha256': sha256_file_hex(path)} for name, path in paths.items()}
    observed_paths = {name: str(path.resolve()) for name, path in paths.items()}
    return {'files': files, 'fingerprint': canonical_json_sha256_hex(files),
            'observed_paths': observed_paths}


def _record_uncommitted_cleanup(shard_root, directory):
    """Durably record every exact unreceipted directory before deleting it."""
    root = Path(shard_root).parent
    files = []
    if directory.is_dir():
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                files.append({'path': str(path.relative_to(root)), 'bytes': path.stat().st_size,
                              'sha256': sha256_file_hex(path)})
    record = {'schema': 'natlang.acquired-session-uncommitted-cleanup/1',
        'directory': str(directory.relative_to(root)), 'recorded_at': datetime.now(timezone.utc).isoformat(),
        'reason': 'unreceipted shard cannot advance source frontier', 'files': files}
    journal = root / 'cleanup-receipt.jsonl'
    with journal.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    dir_fd = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _stop_writer_without_deleting_temp(writer):
    """Quiesce a failed shard compressor while retaining its temp for cleanup proof."""
    if writer.encoder.poll() is None:
        try:
            writer.encoder.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    if writer.encoder.poll() is None:
        writer.encoder.terminate()
    writer.encoder.wait()
    if not writer.out.closed:
        writer.out.flush()
        os.fsync(writer.out.fileno())
        writer.out.close()


def completed_shards(shard_root, binding, expected_sessions):
    """Return a contiguous, content-verified session frontier; reject broken committed receipts."""
    binding_sha = canonical_json_sha256_hex(binding)
    frontier = 0
    verified = []
    shard_root.mkdir(parents=True, exist_ok=True)
    for directory in sorted(shard_root.glob('shard-*')):
        receipt_path = directory / 'receipt.json'
        if not receipt_path.exists():
            _record_uncommitted_cleanup(shard_root, directory)
            shutil.rmtree(directory)
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('schema') != 'natlang.acquired-full-session-shard/1' or receipt.get('binding_sha256') != binding_sha:
            raise ValueError(f'completed session shard has different source/policy/runtime binding: {directory}')
        start, stop = receipt.get('source_record_start'), receipt.get('source_record_stop')
        if (start != frontier or directory.name != f'shard-{start:08d}' or type(stop) is not int
                or stop <= start or stop > expected_sessions or receipt.get('source_record_count') != stop - start):
            raise ValueError(f'completed session shards are not a contiguous in-range prefix: {directory}')
        if {item.get('path') for item in receipt.get('outputs', [])} != {
                'train/native-graphs.jsonl.zst', 'test/native-graphs.jsonl.zst'}:
            raise ValueError(f'completed session shard has an unexpected output set: {directory}')
        output_row_counts = {}
        for output in receipt.get('outputs', []):
            path = directory / output['path']
            if path.stat().st_size != output['bytes'] or sha256_file_hex(path) != output['sha256']:
                raise ValueError(f'completed session shard output differs from its receipt: {path}')
            output_row_counts[output['path']] = output['rows']
        if (output_row_counts['train/native-graphs.jsonl.zst'] != receipt['counts'].get('graphs:train', 0)
                or output_row_counts['test/native-graphs.jsonl.zst'] != receipt['counts'].get('graphs:test', 0)):
            raise ValueError(f'completed session shard row counts differ from its source accounting: {directory}')
        omission = directory / 'omissions.jsonl'
        if sha256_file_hex(omission) != receipt.get('omissions_sha256'):
            raise ValueError(f'completed session shard omission ledger differs from receipt: {omission}')
        verified.append({'path': str(receipt_path), 'sha256': sha256_file_hex(receipt_path),
                         'start': start, 'stop': stop, 'counts': receipt['counts']})
        frontier = stop
    if frontier > expected_sessions:
        raise ValueError('session shard frontier exceeds the source session count')
    return frontier, verified


def _sync_concat(destination, inputs):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output:
            for path in inputs:
                with Path(path).open('rb') as source:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temp_name, destination)
        directory_fd = os.open(destination.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


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


def _tokenizer_rendering_configuration(tokenizer):
    return {'class': type(tokenizer).__module__ + '.' + type(tokenizer).__qualname__,
        'history_reasoning': getattr(tokenizer, 'natlang_history_reasoning', None),
        **{name: getattr(tokenizer, name, None) for name in ('padding_side', 'truncation_side',
            'model_max_length', 'clean_up_tokenization_spaces', 'split_special_tokens')}}


@dataclass(frozen=True)
class PreparedTokenizer:
    """One explicitly bound tokenizer, validated before and after every source."""
    tokenizer: object
    base: Path
    source_fingerprint: str
    expanded_fingerprint: str
    base_files: dict
    rendering_configuration: dict
    render_method: object
    binding_sha256: str

    @property
    def binding(self):
        return {'schema': 'natlang.prepared-native-tokenizer/1',
            'base_files': self.base_files, 'source_fingerprint': self.source_fingerprint,
            'expanded_fingerprint': self.expanded_fingerprint,
            'rendering_configuration': self.rendering_configuration}

    def validate(self, base):
        if canonical_json_sha256_hex(self.binding) != self.binding_sha256:
            raise ValueError('prepared tokenizer binding was mutated')
        if tokenizer_inventory(base) != self.base_files or tokenizer_inventory(self.base) != self.base_files:
            raise ValueError('prepared tokenizer base/configuration inventory changed')
        if tokenizer_fingerprint(self.tokenizer) != self.expanded_fingerprint:
            raise ValueError('prepared tokenizer vocabulary/backend/template was mutated')
        if (_tokenizer_rendering_configuration(self.tokenizer) != self.rendering_configuration
                or self.tokenizer.apply_chat_template != self.render_method):
            raise ValueError('prepared tokenizer rendering configuration was mutated')


def prepare_tokenizer(base):
    base = Path(base).resolve(strict=True)
    files = tokenizer_inventory(base)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    source_fp = tokenizer_fingerprint(tokenizer)
    provision(tokenizer)
    expanded_fp = tokenizer_fingerprint(tokenizer)
    bind_history_reasoning(tokenizer)
    configuration = json.loads(json.dumps(_tokenizer_rendering_configuration(tokenizer)))
    binding = {'schema': 'natlang.prepared-native-tokenizer/1', 'base_files': files,
        'source_fingerprint': source_fp, 'expanded_fingerprint': expanded_fp,
        'rendering_configuration': configuration}
    context = PreparedTokenizer(tokenizer, base, source_fp, expanded_fp, files,
        configuration, tokenizer.apply_chat_template, canonical_json_sha256_hex(binding))
    context.validate(base)
    return context


def main(argv=None, *, prepared=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-receipt', type=Path, required=True)
    parser.add_argument('--base-tokenizer', type=Path, required=True)
    parser.add_argument('--protection', type=Path, required=True)
    parser.add_argument('--decision-hold', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-sessions', type=int, default=100)
    args = parser.parse_args(argv)
    if args.chunk_sessions <= 0:
        raise ValueError('--chunk-sessions must be a positive integer')
    # A callable invocation releases ownership on every success/error path.
    with acquire_output_lease(args.output):
        return _convert(args, prepared)


def _convert(args, prepared):
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
    prepared = prepare_tokenizer(args.base_tokenizer) if prepared is None else prepared
    prepared.validate(args.base_tokenizer)
    tokenizer = prepared.tokenizer
    source_fp, expanded_fp = prepared.source_fingerprint, prepared.expanded_fingerprint
    # Complete source-file held-group closure before writing any shard. Final
    # corpus admission also closes groups across all source files and views.
    held = set(protection['held'])
    source_rows = 0
    for value in sessions(source):
        source_rows += 1
        trajectory = value['trajectory']
        if trajectory['split'] not in ('train', 'test', 'validation'):
            raise ValueError('unknown source session split must not become train')
        if (value['source_file'] != export['contract']['source']
                or value['source_revision'] != export['contract']['revision']):
            raise ValueError('normalized session differs from original export source binding')
        if trajectory['split'] in ('test', 'validation'):
            held.update(trajectory['split_groups'])
    if source_rows != export.get('complete_sessions'):
        raise ValueError('normalized source row count differs from its complete export receipt')
    if (args.output / 'receipt.json').exists():
        raise ValueError('completed graph output is immutable; use its bound receipt')
    runtime = runtime_bindings()
    binding = {'source': {'path': str(source), 'sha256': digest, 'sessions': source_rows},
        'export_receipt': {'path': str(args.export_receipt), 'sha256': sha256_file_hex(args.export_receipt)},
        'source_contract': export['contract'], 'protection': protection['proof'],
        'decision_holds': decision_bindings, 'decision_hold_ids_sha256': canonical_json_sha256_hex(sorted(decision_ids)),
        'source_tokenizer_fingerprint': source_fp, 'expanded_tokenizer_fingerprint': expanded_fp,
        'prepared_tokenizer': prepared.binding,
        'runtime': {'files': runtime['files'], 'fingerprint': runtime['fingerprint']},
        'chunk_sessions': args.chunk_sessions,
        'held_group_closure_sha256': canonical_json_sha256_hex(sorted(held))}
    shard_root = args.output / 'session-shards'
    state_path = args.output / 'resume-state.json'
    previous_state_frontier = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('binding') != binding:
            raise ValueError('resume state is bound to different source, holds, renderer or runtime')
        previous_state_frontier = state.get('completed_source_record_frontier', 0)
    else:
        write_json_durable(state_path, {'schema': 'natlang.acquired-session-resume/1', 'binding': binding,
                                        'binding_sha256': canonical_json_sha256_hex(binding),
                                        'completed_source_record_frontier': 0})
    frontier, shard_receipts = completed_shards(shard_root, binding, source_rows)
    if previous_state_frontier > frontier:
        raise ValueError('resume state claims a source frontier without matching committed shard receipts')
    counts = Counter()
    for item in shard_receipts:
        counts.update(item['counts'])
    if frontier:
        write_json_durable(state_path, {'schema': 'natlang.acquired-session-resume/1', 'binding': binding,
            'binding_sha256': canonical_json_sha256_hex(binding), 'completed_source_record_frontier': frontier,
            'shards': shard_receipts})

    def process(value, omissions, writers, shard_counts):
        trajectory = value['trajectory']; identifier = trajectory['id']; groups = set(trajectory['split_groups'])
        reason = session_exclusion(value, protection, held_sources)
        if reason:
            omissions.write(json.dumps({'id': identifier, 'source_row': value['source_row'], 'reason': reason,
                'original_target_ids': source_target_ids(value),
                'matched_hold_sources': held_session_targets(value, held_sources),
                'hold_matching_scope': 'conservative original source/target identity prefix; no claim of raw-row hash equality',
                'sibling_targets_retained_in_bound_source': reason == 'current-decision-hold'})+'\n')
            shard_counts[reason] += 1
            return
        split = 'test' if groups & held else 'train'
        for graph in session_graphs(value, tokenizer, source_fingerprint=source_fp,
                                    expanded_fingerprint=expanded_fp, split=split):
            writers[split].write(graph)
            shard_counts['graphs:'+split] += 1
            if graph.get('conversion', {}).get('acquired_session', {}).get('view') == 'full-prefix-reasoning-target':
                shard_counts['auxiliary_reasoning_graphs:'+split] += 1
        shard_counts['positive_sessions'] += 1

    sessions_iterator = iter(sessions(source))
    for _ in range(frontier):
        try:
            next(sessions_iterator)
        except StopIteration as exc:
            raise ValueError('resume frontier exceeds actual source records') from exc
    record_index = frontier
    while record_index < source_rows:
        try:
            value = next(sessions_iterator)
        except StopIteration:
            break
        shard_start = record_index
        shard_dir = shard_root / f'shard-{shard_start:08d}'
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True)
        writers = {split: GraphWriter(shard_dir / split / 'native-graphs.jsonl.zst') for split in ('train', 'test')}
        shard_counts = Counter()
        shard_stop = shard_start
        omissions_path = shard_dir / 'omissions.jsonl'
        try:
            with omissions_path.open('w', encoding='utf-8', newline='\n') as omissions:
                while True:
                    process(value, omissions, writers, shard_counts)
                    shard_stop += 1
                    record_index += 1
                    if shard_stop - shard_start >= args.chunk_sessions:
                        break
                    try:
                        value = next(sessions_iterator)
                    except StopIteration:
                        break
                omissions.flush()
                os.fsync(omissions.fileno())
            outputs = [writer.finish() for writer in writers.values()]
        except BaseException:
            for writer in writers.values():
                _stop_writer_without_deleting_temp(writer)
            _record_uncommitted_cleanup(shard_root, shard_dir)
            shutil.rmtree(shard_dir, ignore_errors=True)
            raise
        receipt = {'schema': 'natlang.acquired-full-session-shard/1', 'binding_sha256': canonical_json_sha256_hex(binding),
            'source_record_start': shard_start, 'source_record_stop': shard_stop,
            'source_record_count': shard_stop - shard_start, 'counts': dict(shard_counts),
            'outputs': [{**output, 'path': str(Path(output['path']).relative_to(shard_dir))} for output in outputs],
            'omissions_sha256': sha256_file_hex(omissions_path), 'runtime': runtime,
            'source_sha256': digest, 'protection': protection['proof'], 'decision_holds': decision_bindings,
            'source_tokenizer_fingerprint': source_fp, 'expanded_tokenizer_fingerprint': expanded_fp}
        receipt_path = shard_dir / 'receipt.json'
        write_json_durable(receipt_path, receipt)
        shard_receipts.append({'path': str(receipt_path), 'sha256': sha256_file_hex(receipt_path),
            'start': shard_start, 'stop': shard_stop, 'counts': dict(shard_counts)})
        frontier = shard_stop
        counts.update(shard_counts)
        write_json_durable(state_path, {'schema': 'natlang.acquired-session-resume/1', 'binding': binding,
            'binding_sha256': canonical_json_sha256_hex(binding), 'completed_source_record_frontier': frontier,
            'shards': shard_receipts})
        if frontier != record_index:
            raise AssertionError('durable shard frontier differs from processed input count')
    if frontier != source_rows:
        raise ValueError(f'conversion did not reach the complete source frontier: {frontier}/{source_rows}')
    if sum(counts[key] for key in ('positive_sessions', 'source-reported-non-positive',
            'current-source-quality-or-pending-protection', 'current-decision-hold',
            'source-session-without-assistant-target')) != export['complete_sessions']:
        raise ValueError('complete normalized source session accounting differs')
    if sha256_file_hex(source) != digest:
        raise ValueError('source changed during rendering')
    prepared.validate(args.base_tokenizer)
    final_protection = current_protection(args.protection)
    if final_protection['proof'] != protection['proof']:
        raise ValueError('current source protection changed during rendering')
    if (sorted((Path(__file__).resolve().parents[2] / 'training' / 'decision-holds').glob('*.jsonl')) != current_holds
            or load_decision_holds(current_holds)[1] != decision_bindings
            or canonical_json_sha256_hex(sorted(load_decision_holds(current_holds)[0]))
                != binding['decision_hold_ids_sha256']):
        raise ValueError('current decision holds changed during rendering')
    outputs = []
    for split in ('train', 'test'):
        chunk_paths = [Path(item['path']).parent / split / 'native-graphs.jsonl.zst' for item in shard_receipts]
        output_path = args.output / split / 'native-graphs.jsonl.zst'
        _sync_concat(output_path, chunk_paths)
        expected_rows = sum(output['rows'] for item in shard_receipts
            for output in json.loads(Path(item['path']).read_text())['outputs']
            if output['path'] == f'{split}/native-graphs.jsonl.zst')
        outputs.append({'path': str(output_path), 'rows': expected_rows, 'bytes': output_path.stat().st_size,
                        'sha256': sha256_file_hex(output_path)})
    omissions_path = args.output / 'omissions.jsonl'
    _sync_concat(omissions_path, [Path(item['path']).parent / 'omissions.jsonl' for item in shard_receipts])
    write_json_durable(args.output / 'receipt.json', {'schema': 'natlang.acquired-full-session-graphs/1',
        'source': {'path': str(source), 'sha256': digest},
        'export_receipt': {'path': str(args.export_receipt), 'sha256': sha256_file_hex(args.export_receipt)},
        'protection': protection['proof'], 'decision_holds': decision_bindings,
        'source_tokenizer_fingerprint': source_fp, 'expanded_tokenizer_fingerprint': expanded_fp,
        'builder_sha256': sha256_file_hex(__file__), 'runtime': runtime,
        'prepared_tokenizer': prepared.binding,
        'counts': dict(counts), 'outputs': outputs, 'training_admission': False,
        'source_record_count': source_rows, 'held_group_closure_sha256': binding['held_group_closure_sha256'],
        'shards': shard_receipts, 'resume_state_sha256': sha256_file_hex(state_path),
        'omissions_sha256': sha256_file_hex(omissions_path)})


if __name__ == '__main__':
    main()
