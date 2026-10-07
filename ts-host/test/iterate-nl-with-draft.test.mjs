import assert from 'node:assert/strict';
import test from 'node:test';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';

test('iterateOn passes only the current draft to an nl.with<Draft> child and receives a Draft back', async () => {
  const requests = [];
  const code = `
type Draft = { train: string; membrane: string; pressure: string; status: string };
type Progress = { pass: number; draft: Draft };
const revise = async (state: Progress): Promise<Progress> => {
  const update = nl.with<Draft>({ passName: "train register" })
    \`Apply the captured pass to the supplied full draft and return the complete Draft.\`;
  const nextDraft = await update(state.draft);
  return { pass: state.pass + 1, draft: nextDraft };
};
const initial: Progress = { pass: 0, draft: { train: "T1", membrane: "M1", pressure: "1 kPa", status: "return" } };
const final = await iterateOn(revise, initial)
  .checkProgress('off').withLimit({ maxSteps: 1 }).until(state => state.pass === 1);
return final.draft;
`;
  const driver = async request => {
    requests.push(request);
    if (requests.length === 1) return { calls: [['eval', { code }]] };
    return { calls: [['return_result', { status: 'success', value: {
      train: 'DESAL-TRAIN-5', membrane: 'BANK-M5-C', pressure: '1 kPa', status: 'return',
    } }]] };
  };
  const runtime = createNatlangRuntime({ model: driver });
  const root = defineNatlang('---\nargs: {}\nreturns: "{ train: string, membrane: string, pressure: string, status: string }"\n---\nRun one typed draft revision through iterateOn.\n', { name: 'root' });
  const actual = await runtime.run(() => root());

  assert.deepEqual(actual, { train: 'DESAL-TRAIN-5', membrane: 'BANK-M5-C', pressure: '1 kPa', status: 'return' });
  assert.ok(requests.length >= 2, 'the root eval and inline child call both reached the model');
  const childOpening = String(requests[1].messages[1].content);
  assert.match(childOpening, /nl@eval:5\(input: Draft\): Draft/);
  assert.match(childOpening, /In eval you can use input, passName/);
});
