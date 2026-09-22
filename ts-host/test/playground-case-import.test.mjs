import assert from 'node:assert/strict';
import test from 'node:test';
import { convertCase } from '../scripts/import-playground-cases.mjs';

const baseCase = () => ({ schema: 'natlang.playground.case/1', id: 'accepted-1', revision: 3,
  reviewStatus: 'accepted', admission: { admitted: true }, split: 'train',
  source: { root: 'main.nl', files: { 'main.nl': `---\ndescription: return supplied text\nargs:\n  value: string\nreturns: string\n---\nReturn the supplied value.\n` } },
  expected: { kind: 'done', value: 'hello' },
  trace: [
    { seq: 0, version: 'reduction-trace/1', kind: 'manifest' },
    { seq: 1, version: 'reduction-trace/1', kind: 'action', call_id: '$root@1', outcome: 'ok',
      name: 'write', arguments: { path: 'return', type: 'string', value: 'hello' } },
    { seq: 2, version: 'reduction-trace/1', kind: 'state', phase: 'final', outcome: 'done', value: 'hello' },
  ] });

test('Node playground importer converts admitted root write cases to lambda_scenario IR', async () => {
  const result = await convertCase(baseCase());
  assert.equal(result.version, 'natlang.program/1');
  assert.equal(result.kind, 'lambda_scenario');
  assert.equal(result.semantics.root.$lambda.instructions, 'Return the supplied value.\n');
  assert.deepEqual(result.semantics.root.$lambda.args, {});
  assert.deepEqual(result.semantics.operations, [
    { op: 'assign', target: 'return', value_type: 'string', value: 'hello' },
  ]);
  assert.deepEqual(result.semantics.contract.required_actions, [
    { tool: 'write', arguments: { path: 'return', type: 'string', value: 'hello' } },
  ]);
});

test('Node playground importer rejects non-admitted, discontinuous, effectful, and linked cases', async () => {
  const notAdmitted = baseCase(); notAdmitted.admission.admitted = false;
  await assert.rejects(convertCase(notAdmitted), /not reviewed and exactly admitted/);
  const discontinuous = baseCase(); discontinuous.trace[1].seq = 8;
  await assert.rejects(convertCase(discontinuous), /discontinuous reduction trace/);
  const effectful = baseCase(); effectful.trace.splice(2, 0,
    { seq: 2, version: 'reduction-trace/1', kind: 'effect' });
  effectful.trace[3].seq = 3;
  await assert.rejects(convertCase(effectful), /host effects need a richer adapter/);
  const linked = baseCase(); linked.source.files['main/child.nl'] = '---\nreturns: string\n---\nDone.\n';
  await assert.rejects(convertCase(linked), /linked child functions need a graph adapter/);
});
