import assert from 'node:assert/strict';
import { test } from 'node:test';
import { executeProgram, validateFocusedRecord } from '../dist/teacher/collector.js';

const record = () => ({ version: 'natlang.program/2', id: 'controller-fixture', kind: 'lambda_source',
  semantics: {
    root: 'entry.nl',
    files: {
      'natlang.json': '{}',
      'entry.nl': '---\nargs:\n  value: number\nreturns: number\nuses: [workflow]\n---\nRun the application controller.\n',
      'entry/advance.nl': '---\nargs:\n  value: number\nreturns: number\n---\nAdvance the workflow.\n',
      'entry/advance/step.nl': '---\nargs:\n  value: number\nreturns: number\n---\nSelect the transformed value.\n',
      'workflow.ts': 'export async function run(value: number, operations: { step(value: number): Promise<number> }): Promise<number> { return (await operations.step(value)) + 1; }\n',
    },
    inputs: { value: 4 }, expected: 9, oracle: 'exact',
    services: { probe: 'export function read(): number { return 4; }' },
    service_scopes: { probe: ['entry/advance/step.nl'] },
    application_controller: { schema: 'natlang.application-controller/1', module: 'workflow', export: 'run',
      operations: 'advance', input_arguments: ['value'] },
  },
});

test('collector invokes the declared TypeScript controller and retains its Natlang child graph', async () => {
  const run = await executeProgram(record(), async () => ({
    calls: [['eval', { code: 'return value + await probe.read();', finish: true }]],
  }), { rootSeed: 5, systemPrompt: '', contextTokens: 4096, maxTurns: 3,
    runId: 'application-controller-fixture' });
  assert.equal(run.outcome.value, 9);
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome.rejection_reasons));
  assert.deepEqual(run.trace.filter(event => event.kind === 'application_controller_call').map(event => event.phase),
    ['start', 'completed']);
  const controller = run.trace.find(event => event.kind === 'application_controller_call' && event.phase === 'completed');
  assert.ok(controller.child_invocation_count > 0);
  assert.ok(run.invocationTraces?.length > 0);
  assert.ok(run.outcome.execution_graph.some(event => event.definition?.id === 'nl:entry/advance/step.nl'));
});

test('collector rejects missing or detached application controller bindings before execution', () => {
  const missingInput = record();
  missingInput.semantics.application_controller.input_arguments = ['absent'];
  assert.throws(() => validateFocusedRecord(missingInput), /bound input is absent/);
  const detached = record();
  detached.semantics.files['entry.nl'] = detached.semantics.files['entry.nl'].replace('uses: [workflow]\n', '');
  assert.throws(() => validateFocusedRecord(detached), /absent from the task package/);
});

test('collector rejects a controller that completes without invoking its declared Natlang operation', async () => {
  const detached = record();
  detached.semantics.files['workflow.ts'] =
    'export async function run(value: number): Promise<number> { return value + 5; }\n';
  const run = await executeProgram(detached, async () => ({ calls: [] }), { rootSeed: 5, systemPrompt: '',
    contextTokens: 4096, maxTurns: 3, runId: 'application-controller-detached-fixture' });
  assert.equal(run.outcome.value, 9);
  assert.equal(run.outcome.accepted, false);
  assert.ok(run.outcome.rejection_reasons.includes('application_controller'));
});
