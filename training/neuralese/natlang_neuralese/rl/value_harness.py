"""A real-model protocol fixture through the unchanged native collector.

Forced tool decisions isolate transport/typing/graph correctness; these are NOT
student task-success examples and must never enter RL rewards or training data.
The writer vectors themselves run on the real model and checkpoint projection.
"""
from __future__ import annotations
import json
import subprocess
import threading
from pathlib import Path


def validate(engine, out, runtime):
    from ..serve.http import serve
    from ..common.jsonio import write_json_durable
    out, runtime = Path(out), Path(runtime)
    out.mkdir(parents=True, exist_ok=True)
    code = ('const seed: Neuralese<string> = await nl<Neuralese<string>>`Write a note.`(); '
            'const reader: Neuralese<(prior: Neuralese<string>) => Promise<string>> = '
            'nl.with<string>({})`Read the note and answer ok.`; return await reader(seed);')
    plans = [
        ["<|tool_call_start|>[eval(code=" + repr(code) + ", finish=True)]<|tool_call_end|>"],
        ["<|tool_call_start|>[eval(code='return ", {'neuralese': 'write'}, ";', finish=True)]<|tool_call_end|>"],
        ["<|tool_call_start|>[return_result(status='success', value='ok')]<|tool_call_end|>"],
    ]
    original, captured = engine.submit, []
    def submit(request):
        if len(captured) >= len(plans):
            raise RuntimeError('unexpected extra request in forced value harness fixture')
        request.forced = plans[len(captured)]
        request.guidance = None
        request.rl_capture = True
        request.neuralese_length = 2
        request.max_tokens = 1024
        captured.append(request)
        future = original(request)
        def diagnose(done):
            if done.exception() is not None:
                import traceback
                traceback.print_exception(done.exception())
        future.add_done_callback(diagnose)
        return future
    engine.submit = submit
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
        from ..serve.store import encode_block
        save_bundle([{'turns': rollout_turns(row), 'execution_graph': graph}], out / 'blocks',
                    policy_id='diagnostic-not-training', fetch=lambda bid: encode_block(engine.lookup(bid)))
        report = {'schema': 'natlang.rl-value-harness-validation/1', 'accepted': True,
            'forced_tool_decisions': True, 'admitted_training_or_reward': False,
            'real_model_writer': True, 'model_requests': len(captured),
            'writer_nodes': len(writes), 'reader_nodes': len(reads),
            'durable_bundle': str(out / 'blocks/manifest.json')}
        write_json_durable(out / 'validation.json', report)
        return report
    finally:
        server.shutdown()
        server.server_close()
        engine.stop()
        engine.submit = original
