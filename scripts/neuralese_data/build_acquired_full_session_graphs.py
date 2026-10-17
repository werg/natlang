"""Complete source-curated sessions through existing serving/native renderers."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
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
from natlang_neuralese.serve.chat import (RequestError, bind_history_reasoning,
    normalize_tool_call_arguments)
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter, numerical_runtime
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease

_SESSION_GRAPH_RENDER_AST_SHA256 = '7b73bf21659f50fbee4c8b990b0ab0895c7d880d25a368b70986ba1d0c8a76bc'
_SESSION_GRAPH_PREVALIDATION_PARENT_BUILDER_SHA256 = 'cf4346e17dcc2c9f4770e2d8fb142cf597563e6551b8d51329c5c224ba5aaf1b'
_SESSION_GRAPH_CONTINUATION_PROOF = Path(__file__).resolve().with_name(
    'acquired_session_render_pending_continuation_20261017.json')


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


def prevalidate_session_tool_arguments(value, *, template_backbone=None):
    """Validate source arguments against the exact tokenizer template contract.

    The serving normalizer decodes JSON once.  LFM2's chat template additionally
    requires tool arguments to be mappings; Mellum accepts scalar JSON values.
    Keep that distinction tied to the prepared tokenizer receipt, never the
    source name or output directory.
    """
    normalized, conversions = normalize_tool_call_arguments(value['trajectory']['messages'])
    if template_backbone == 'lfm2':
        for message_index, message in enumerate(normalized):
            for call_index, call in enumerate(message.get('tool_calls') or []):
                function = call.get('function', call)
                if 'arguments' in function and not isinstance(function['arguments'], dict):
                    raise RequestError('tool-arguments',
                        f'tool-call arguments at messages/{message_index}/tool_calls/{call_index} '
                        'must decode to a mapping for the lfm2 tokenizer template')
    return normalized, conversions


def source_render_pending_omission(value, error):
    """Describe only a proven serving tool-argument parse failure for this source row."""
    if not isinstance(error, RequestError) or error.code != 'tool-arguments':
        raise error
    return {'id': value['trajectory']['id'], 'source_row': value['source_row'],
        'reason': 'source-render-pending:tool-arguments',
        'pending_status': 'source-faithful-review-pending',
        'normalized_source_sha256': canonical_json_sha256_hex(value),
        'original_source_file': value['source_file'],
        'original_target_ids': source_target_ids(value), 'error_code': error.code}


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
        'conversion_output_lease': root / 'scripts' / 'neuralese_data' / 'conversion_output_lease.py',
        'split_group_closure': root / 'scripts' / 'neuralese_data' / 'splits.py',
        'render_pending_continuation': root / 'scripts' / 'neuralese_data' /
            'acquired_session_render_pending_continuation_20261017.json',
    }
    # Identity follows the logical runtime files and their bytes, not checkout
    # location. Keep observed paths as receipt metadata outside the fingerprint.
    files = {name: {'sha256': sha256_file_hex(path)} for name, path in paths.items()}
    observed_paths = {name: str(path.resolve()) for name, path in paths.items()}
    return {'files': files, 'fingerprint': canonical_json_sha256_hex(files),
            'observed_paths': observed_paths, 'numerical_runtime': numerical_runtime()}


def prove_render_pending_runtime_continuation(parent_binding, current_binding, parent_frontier):
    """Prove the one reviewed builder-only change can resume prior render shards."""
    if type(parent_frontier) is not int or parent_frontier < 0:
        raise ValueError('runtime continuation has an invalid prior source frontier')
    old_runtime = parent_binding.get('runtime', {})
    new_runtime = current_binding.get('runtime', {})
    old_files, new_files = old_runtime.get('files', {}), new_runtime.get('files', {})
    if set(old_files) | {'render_pending_continuation'} != set(new_files) or 'builder' not in old_files:
        raise ValueError('runtime continuation changes the declared dependency set')
    if old_files['builder'].get('sha256') != _SESSION_GRAPH_PREVALIDATION_PARENT_BUILDER_SHA256:
        raise ValueError('runtime continuation parent builder is not the reviewed renderer')
    if any(old_files[name] != new_files[name] for name in old_files if name != 'builder'):
        raise ValueError('runtime continuation changes a renderer dependency')
    proof_path = _SESSION_GRAPH_CONTINUATION_PROOF
    try:
        proof_bytes = proof_path.read_bytes()
        proof = json.loads(proof_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('runtime continuation proof file is absent or invalid') from exc
    if (new_files['render_pending_continuation'].get('sha256') != hashlib.sha256(proof_bytes).hexdigest()
            or proof != {'schema': 'natlang.acquired-session-render-pending-runtime-continuation-proof/1',
                'parent_builder_sha256': _SESSION_GRAPH_PREVALIDATION_PARENT_BUILDER_SHA256,
                'current_builder_sha256': new_files['builder'].get('sha256'),
                'unchanged_session_graph_renderer_ast_sha256': _SESSION_GRAPH_RENDER_AST_SHA256,
                'allowed_delta': 'pre-render serving tool-argument validation and explicit pending-source accounting'}):
        raise ValueError('runtime continuation proof does not bind the exact code transition')
    if old_runtime.get('fingerprint') != canonical_json_sha256_hex(old_files):
        raise ValueError('runtime continuation parent fingerprint is invalid')
    if new_runtime.get('fingerprint') != canonical_json_sha256_hex(new_files):
        raise ValueError('runtime continuation current fingerprint is invalid')
    if current_binding.get('runtime') != {'files': new_files, 'fingerprint': new_runtime['fingerprint']}:
        raise ValueError('runtime continuation current binding has unexpected runtime metadata')
    for name in set(parent_binding) | set(current_binding):
        if name != 'runtime' and parent_binding.get(name) != current_binding.get(name):
            raise ValueError('runtime continuation changes source, protection, tokenizer or chunk policy')
    try:
        tree = ast.parse(Path(__file__).read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == 'session_graphs')
    except (OSError, SyntaxError, StopIteration) as exc:
        raise ValueError('runtime continuation cannot inspect the current renderer') from exc
    function.body = [node for node in function.body if not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'prevalidate_session_tool_arguments')]
    render_ast_sha = hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest()
    if render_ast_sha != _SESSION_GRAPH_RENDER_AST_SHA256:
        raise ValueError('runtime continuation changed the whole-session graph renderer')
    return {'schema': 'natlang.acquired-session-render-pending-runtime-continuation/1',
        'parent_binding_sha256': canonical_json_sha256_hex(parent_binding),
        'current_binding_sha256': canonical_json_sha256_hex(current_binding),
        'parent_frontier': parent_frontier,
        'parent_builder_sha256': old_files['builder']['sha256'],
        'current_builder_sha256': new_files['builder']['sha256'],
        'unchanged_session_graph_renderer_ast_sha256': render_ast_sha,
        'allowed_delta': 'pre-render serving tool-argument validation and explicit pending-source accounting'}


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


def completed_shards(shard_root, binding, expected_sessions, *,
                     schema='natlang.acquired-full-session-shard/1',
                     additional_outputs=(), compatible_bindings=None):
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
        receipt_binding = receipt.get('binding_sha256')
        compatible_bindings = compatible_bindings or {}
        compatible_frontier = compatible_bindings.get(receipt_binding)
        if (receipt.get('schema') != schema or (receipt_binding != binding_sha
                and (compatible_frontier is None or receipt.get('source_record_stop', expected_sessions + 1)
                     > compatible_frontier))):
            raise ValueError(f'completed session shard has different source/policy/runtime binding: {directory}')
        start, stop = receipt.get('source_record_start'), receipt.get('source_record_stop')
        if (start != frontier or directory.name != f'shard-{start:08d}' or type(stop) is not int
                or stop <= start or stop > expected_sessions or receipt.get('source_record_count') != stop - start):
            raise ValueError(f'completed session shards are not a contiguous in-range prefix: {directory}')
        if {item.get('path') for item in receipt.get('outputs', [])} != {
                'train/native-graphs.jsonl.zst', 'test/native-graphs.jsonl.zst', *additional_outputs}:
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
    # Validate every tool call before the first graph can be yielded. This is
    # the same exact validator used by serving; the returned copy is discarded.
    prevalidate_session_tool_arguments(value,
        template_backbone=getattr(tokenizer, 'natlang_history_reasoning', {}).get('backbone'))
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
    template_backbone = prepared.rendering_configuration.get('history_reasoning', {}).get('backbone')
    if template_backbone not in ('lfm2', 'mellum'):
        raise ValueError('prepared tokenizer receipt lacks a supported template backbone')
    # Complete source-file held-group closure before writing any shard. Final
    # corpus admission also closes groups across all source files and views.
    held = set(protection['held'])
    source_groups = []
    source_rows = 0
    for value in sessions(source):
        source_rows += 1
        trajectory = value['trajectory']
        source_groups.append(trajectory['split_groups'])
        if trajectory['split'] not in ('train', 'test', 'validation'):
            raise ValueError('unknown source session split must not become train')
        if (value['source_file'] != export['contract']['source']
                or value['source_revision'] != export['contract']['revision']):
            raise ValueError('normalized session differs from original export source binding')
        if trajectory['split'] in ('test', 'validation'):
            held.update(trajectory['split_groups'])
    from scripts.neuralese_data.splits import close_group_sets
    closed, _ = close_group_sets(source_groups, {**{key: protection[key]
        for key in ('quality', 'pending')}, 'held': held})
    protection = {**protection, **closed}
    held = closed['held']
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
    runtime_continuation = None
    compatible_bindings = {}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get('binding') != binding:
            parent_binding = state.get('binding')
            previous_state_frontier = state.get('completed_source_record_frontier', 0)
            runtime_continuation = prove_render_pending_runtime_continuation(
                parent_binding, binding, previous_state_frontier)
            # Existing output receipts remain byte-exact and retain their old
            # binding. Prove that this source prefix has no newly-classified
            # malformed tool calls before allowing the reviewed continuation.
            for index, value in enumerate(sessions(source)):
                if index >= previous_state_frontier:
                    break
                if session_exclusion(value, protection, held_sources):
                    continue
                try:
                    prevalidate_session_tool_arguments(value, template_backbone=template_backbone)
                except RequestError as error:
                    if error.code == 'tool-arguments':
                        raise ValueError('prior committed source prefix contains a newly render-pending row') from error
                    raise
            compatible_bindings[canonical_json_sha256_hex(parent_binding)] = previous_state_frontier
            state = {'schema': 'natlang.acquired-session-resume/1', 'binding': binding,
                'binding_sha256': canonical_json_sha256_hex(binding),
                'completed_source_record_frontier': previous_state_frontier,
                'shards': state.get('shards', []), 'parent_binding': parent_binding,
                'runtime_continuation': runtime_continuation}
            write_json_durable(state_path, state)
        previous_state_frontier = state.get('completed_source_record_frontier', 0)
    else:
        state = {'schema': 'natlang.acquired-session-resume/1', 'binding': binding,
                                        'binding_sha256': canonical_json_sha256_hex(binding),
                                        'completed_source_record_frontier': 0}
        write_json_durable(state_path, state)
    if state_path.exists() and state.get('runtime_continuation'):
        runtime_continuation = state['runtime_continuation']
        parent_binding = state.get('parent_binding')
        if parent_binding is None or runtime_continuation != prove_render_pending_runtime_continuation(
                parent_binding, binding, runtime_continuation.get('parent_frontier')):
            raise ValueError('resume state runtime continuation witness is invalid')
        compatible_bindings[canonical_json_sha256_hex(parent_binding)] = runtime_continuation['parent_frontier']
    frontier, shard_receipts = completed_shards(shard_root, binding, source_rows,
        compatible_bindings=compatible_bindings)
    if previous_state_frontier > frontier:
        raise ValueError('resume state claims a source frontier without matching committed shard receipts')
    counts = Counter()
    for item in shard_receipts:
        counts.update(item['counts'])
    if frontier:
        write_json_durable(state_path, {'schema': 'natlang.acquired-session-resume/1', 'binding': binding,
            'binding_sha256': canonical_json_sha256_hex(binding), 'completed_source_record_frontier': frontier,
            'shards': shard_receipts, **({'parent_binding': state['parent_binding'],
                'runtime_continuation': runtime_continuation} if runtime_continuation else {})})

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
        try:
            prevalidate_session_tool_arguments(value, template_backbone=template_backbone)
        except RequestError as error:
            pending = source_render_pending_omission(value, error)
            omissions.write(json.dumps(pending, ensure_ascii=False, sort_keys=True) + '\n')
            shard_counts[pending['reason']] += 1
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
            'omissions_sha256': sha256_file_hex(omissions_path),
            'runtime': {**runtime, 'numerical_runtime': numerical_runtime()},
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
            'shards': shard_receipts, **({'parent_binding': state['parent_binding'],
                'runtime_continuation': runtime_continuation} if runtime_continuation else {})})
        if frontier != record_index:
            raise AssertionError('durable shard frontier differs from processed input count')
    if frontier != source_rows:
        raise ValueError(f'conversion did not reach the complete source frontier: {frontier}/{source_rows}')
    if sum(counts[key] for key in ('positive_sessions', 'source-reported-non-positive',
            'current-source-quality-or-pending-protection', 'current-decision-hold',
            'source-session-without-assistant-target', 'source-render-pending:tool-arguments')) != export['complete_sessions']:
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
        'shards': shard_receipts, 'runtime_continuation': runtime_continuation,
        'resume_state_sha256': sha256_file_hex(state_path),
        'omissions_sha256': sha256_file_hex(omissions_path)})


if __name__ == '__main__':
    main()
