"""A real-model protocol fixture through the native collector and positional-state writer."""
from __future__ import annotations
import json
import subprocess
import threading
from pathlib import Path


def native_artifact_bytes(engine, artifact_id):
    """Fetch the exact native artifact produced by the protocol fixture."""
    if artifact_id.startswith('nzs1_'):
        from ..serve.store import encode_native_state
        record = engine.store.get_native_state(artifact_id)
        if record is None:
            raise ValueError(f'value harness omitted native state {artifact_id}')
        return encode_native_state(record.block)
    if artifact_id.startswith('nnt1_'):
        from ..model.native_artifact import encode_native_tensor_artifact
        record = engine.store.get_native_tensor_artifact(artifact_id)
        if record is None:
            raise ValueError(f'value harness omitted native tensor artifact {artifact_id}')
        return encode_native_tensor_artifact(record.artifact)
    raise ValueError(f'value harness received a non-native artifact ID: {artifact_id}')


def _forced_native_reply_tokens(engine, segments):
    """Encode a diagnostic assistant reply, writing named spans through native role markers."""
    result = []
    for kind, text in segments:
        if kind == 'text':
            result.extend(engine._template_tokens(text))
        elif kind == 'native-state':
            result.extend((engine.role_tokens.produce_open, *engine._tokens(text), engine.role_tokens.produce_close))
        else:
            raise ValueError(f'unknown native value-harness reply segment: {kind}')
    result.append(engine.stop_ids[0])
    return result


def validate(engine, out, runtime):
    from ..serve.http import serve
    from ..common.jsonio import write_json_durable
    out, runtime = Path(out), Path(runtime)
    out.mkdir(parents=True, exist_ok=True)
    code = ('const seed: Neuralese<string> = await nl<Neuralese<string>>`Write a note.`(); '
            'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = '
            'nl.with<string>({})`Read the note and answer ok.`; return await reader(seed);')
    replies = [
        [('text', "<|tool_call_start|>[eval(code=" + repr(code) + ", finish=True)]<|tool_call_end|>")],
        [('text', "<|tool_call_start|>[eval(code='return "),
         ('native-state', 'A durable note for the reader.'),
         ('text', ";', finish=True)]<|tool_call_end|>")],
        [('text', "<|tool_call_start|>[return_result(status='success', value='ok')]<|tool_call_end|>")],
    ]
    original, original_token, captured = engine.submit, engine._native_token, []
    def submit(request):
        if len(captured) >= len(replies):
            raise RuntimeError('unexpected extra request in forced value harness fixture')
        request.value_harness_tokens = _forced_native_reply_tokens(engine, replies[len(captured)])
        request.guidance = None
        request.rl_capture = True
        request.native_recording = True
        request.temperature = 0.0
        request.max_tokens = 1024
        captured.append(request)
        future = original(request)
        def diagnose(done):
            if done.exception() is not None:
                import traceback
                traceback.print_exception(done.exception())
        future.add_done_callback(diagnose)
        return future
    def native_token(sequence):
        plan = sequence.request.value_harness_tokens
        if not plan:
            raise RuntimeError('forced native value-harness reply ended without a stop token')
        return plan.pop(0), None
    engine.submit = submit
    engine._native_token = native_token
    engine.start()
    server = serve(engine, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    script = r'''
import { pathToFileURL } from 'node:url';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const config = JSON.parse(input);
const load = name => import(pathToFileURL(config.runtime + '/dist/' + name).href);
const { definitionProject } = await load('teacher/program.js');
const { nativeJobRunner, expectedProvenance, defaultToolSurfaceHash } = await load('teacher/collector.js');
const project = definitionProject('valueflow', { returns: 'string', instructions: 'Pass a note to a reader.' });
const record = { version: 'natlang.program/2', id: 'diagnostic-forced-value-flow', kind: 'lambda_source',
 source: 'diagnostic-not-training', split: 'train', source_ids: ['diagnostic-forced-value-flow'],
 source_groups: ['diagnostic-not-training'], license: 'test', semantics: {
 root: project.root, files: project.files, inputs: {}, expected: 'ok', operation: 'exact' } };
const options = { jobs: config.out + '/jobs', output: config.out + '/result.jsonl', workers: 1,
 modelId: 'diagnostic-forced-value-flow', endpoint: config.endpoint, rootSeed: 42, systemPrompt: 'Test.',
 contextTokens: 8192, toolSurfaceSha256: await defaultToolSurfaceHash(), maxTurns: 4,
 collectionRole: 'student', neuraleseServer: true,
 request: { guidance: false, x_natlang_rl_capture: true } };
await mkdir(options.jobs, { recursive: true });
const row = await nativeJobRunner(options)({ index: 0, record }, expectedProvenance(record, options));
await writeFile(options.output, JSON.stringify(row) + '\n');
if (!row.outcome.accepted) throw new Error(JSON.stringify(row.outcome));
'''
    try:
        command = subprocess.run(['node', '--input-type=module', '--eval', script], input=json.dumps({
            'out': str(out), 'runtime': str(runtime), 'endpoint': f'http://127.0.0.1:{server.server_address[1]}'}),
            text=True, capture_output=True)
        (out / 'node.log').write_text(command.stdout + command.stderr)
        if command.returncode:
            raise RuntimeError('real value harness failed; see ' + str(out / 'node.log'))
        row = json.loads((out / 'result.jsonl').read_text())
        graph = row['outcome']['execution_graph']
        writes = [node for node in graph if node['kind'] == 'block_write']
        reads = [node for node in graph if node['kind'] == 'block_read']
        if not writes or not any(edge.get('node') == write['node'] for read in reads
                                for edge in read.get('inputs', []) for write in writes):
            raise RuntimeError('real value harness lost exact writer-reader occurrence edge')
        from .task_rollouts import rollout_turns
        from .graph_replay import save_bundle
        save_bundle([{'turns': rollout_turns(row), 'execution_graph': graph}], out / 'blocks',
                    policy_id='diagnostic-not-training', fetch=lambda bid: native_artifact_bytes(engine, bid))
        report = {'schema': 'natlang.rl-value-harness-validation/1', 'accepted': True,
            'forced_tool_decisions': True, 'admitted_training_or_reward': False,
            'forced_native_write': True, 'checkpoint_native_writer_compute': True,
            'model_requests': len(captured),
            'writer_nodes': len(writes), 'reader_nodes': len(reads),
            'durable_bundle': str(out / 'blocks/manifest.json')}
        write_json_durable(out / 'validation.json', report)
        return report
    finally:
        server.shutdown()
        server.server_close()
        engine.stop()
        engine.submit = original
        engine._native_token = original_token
