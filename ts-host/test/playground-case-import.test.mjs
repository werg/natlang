import assert from 'node:assert/strict';
import test from 'node:test';
import { convertCase } from '../scripts/import-playground-cases.mjs';

const baseCase = () => ({ schema: 'natlang.playground.case/1', id: 'accepted-1', revision: 3,
  reviewStatus: 'accepted', admission: { admitted: true }, split: 'train', inputs: { value: 'hello' },
  source: { root: 'main.nl', files: { 'main.nl': `---\ndescription: return supplied text\nargs:\n  value: string\nreturns: string\n---\nReturn the supplied value.\n` } },
  expected: { kind: 'done', value: 'hello' },
  trace: [
    { seq: 0, version: 'reduction-trace/1', kind: 'manifest' },
    { seq: 1, version: 'reduction-trace/1', kind: 'action', name: 'return_result', arguments: { value: 'hello' } },
    { seq: 2, version: 'reduction-trace/1', kind: 'state', phase: 'final', outcome: 'done', value: 'hello' },
  ] });

test('the playground importer converts an admitted case into a program IR project', async () => {
  const result = await convertCase(baseCase());
  assert.equal(result.version, 'natlang.program/2');
  assert.equal(result.kind, 'lambda_source');
  assert.equal(result.semantics.root, 'main.nl');
  assert.deepEqual(result.semantics.inputs, { value: 'hello' });
  assert.equal(result.semantics.expected, 'hello');
  assert.equal(Object.hasOwn(result.semantics.contract, 'required_actions'), false, 'no tool calls in program IR');
  const linked = baseCase();
  linked.source.files['main/child.nl'] = '---\nargs: {}\nreturns: string\n---\nDone.\n';
  assert.deepEqual(Object.keys((await convertCase(linked)).semantics.files).sort(), ['main.nl', 'main/child.nl']);
});

test('the playground importer rejects non-admitted, discontinuous, effectful, and mismatched cases', async () => {
  const notAdmitted = baseCase(); notAdmitted.admission.admitted = false;
  await assert.rejects(convertCase(notAdmitted), /not reviewed and exactly admitted/);
  const discontinuous = baseCase(); discontinuous.trace[1].seq = 8;
  await assert.rejects(convertCase(discontinuous), /discontinuous reduction trace/);
  const effectful = baseCase(); effectful.trace.splice(2, 0, { seq: 2, version: 'reduction-trace/1', kind: 'effect' });
  effectful.trace.forEach((event, index) => { event.seq = index; });
  await assert.rejects(convertCase(effectful), /effect contract/);
  const unknownInput = baseCase(); unknownInput.inputs = { other: 1 };
  await assert.rejects(convertCase(unknownInput), /not a parameter/);
});
