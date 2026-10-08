import test from 'node:test';
import assert from 'node:assert/strict';
const collectorModule = process.env.NATLANG_COLLECTOR_MODULE ?? '../dist/teacher/collector.js';
const { collectedInvocationTraceEvents } = await import(collectorModule);

test('completed collector trace export tags root and child events without merging sequence spaces', () => {
  const root = [{ kind: 'manifest', seq: 0, run_id: 'root-run' },
    { kind: 'iteration_step', seq: 1, node: 'root-run#1', inputs: [] }];
  const children = [
    { callId: 'root-run/1', events: [{ kind: 'invocation', seq: 0, call_id: 'root-run/1' },
      { kind: 'block_write', seq: 3, call_id: 'root-run/1', block: 'nz1_example' }] },
    { callId: 'root-run/2', events: [{ kind: 'invocation', seq: 0, call_id: 'root-run/2' }] },
  ];

  const output = collectedInvocationTraceEvents('root-run', root, children);

  assert.deepEqual(output.map(({ trace_role, invocation_id, seq }) => [trace_role, invocation_id, seq]), [
    ['root', 'root-run', 0], ['root', 'root-run', 1],
    ['child', 'root-run/1', 0], ['child', 'root-run/1', 3], ['child', 'root-run/2', 0],
  ]);
  assert.deepEqual(output[3], { kind: 'block_write', seq: 3, call_id: 'root-run/1', block: 'nz1_example',
    trace_role: 'child', invocation_id: 'root-run/1' });
  assert.equal(root[0].trace_role, undefined);
  assert.equal(children[0].events[0].invocation_id, undefined);
});
