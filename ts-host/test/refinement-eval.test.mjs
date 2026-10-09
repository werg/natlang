import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { modelTurnsSoFar } from '../dist/native/agent.js';

const POLITE = 'a reply that is polite';
const REPLY = `---\nargs: { complaint: string }\nreturns: 'Is<string, "${POLITE}">'\n---\nAnswer the complaint.\n`;

/** A model that plays `steps` (one eval per turn after the runtime's own first eval) and a judge that dislikes "fault". */
function scenario(steps, returns = REPLY) {
  const judged = [], seen = [];
  const driver = Object.assign(async ({ messages }) => {
    const turn = modelTurnsSoFar(messages);
    seen.push(String(messages.at(-1).content));
    const step = steps[Math.min(turn, steps.length - 1)];
    return { calls: [step.tool ? [step.tool, step.args] : ['eval', step]] };
  }, { decide: async request => {
    const value = /<<<value\n([^]*?)\nvalue>>>/.exec(String(request.messages.at(-1).content))[1];
    judged.push(value);
    const p = /fault/.test(value) ? 0.02 : 0.98;
    return { log_probs: [Math.log(p), Math.log(1 - p)] };
  } });
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, calls: false, trace: trace => traces.push(trace) });
  const fn = loadVirtualNatlang({ 'reply.nl': returns }, 'reply.nl');
  return { judged, seen, traces, run: () => runtime.run(() => fn('late')) };
}

test('an eval that finishes with an unsatisfying refined result is rejected and the model repairs it', async () => {
  const s = scenario([{ code: 'return "your fault";', finish: true }, { code: 'return "Sorry about that.";', finish: true }]);
  assert.equal(await s.run(), 'Sorry about that.');
  assert.ok(s.seen.some(text => /refinement-unsatisfied: Revise the value at return so that it is/.test(text)), s.seen.join('\n---\n'));
});

test('a staged eval return is judged before it is kept', async () => {
  const s = scenario([{ code: 'return "your fault";' }, { code: 'return "Sorry about that.";' }, { tool: 'return_result', args: { status: 'success' } }]);
  await s.run().catch(() => {});
  assert.ok(s.judged.includes('your fault'));
  assert.ok(s.seen.some(text => /This is not the result: refinement-unsatisfied/.test(text)), s.seen.join('\n---\n'));
});

test('a refined local is checked when the model writes it', async () => {
  const s = scenario([{ code: `let note: Is<string, "${POLITE}"> = "your fault";` }, { code: 'return "Fine.";', finish: true }]);
  assert.equal(await s.run(), 'Fine.');
  assert.ok(s.seen.some(text => /refinement-unsatisfied/.test(text)), s.seen.join('\n---\n'));
});

test('refine and assume are bound in the eval scope', async () => {
  const s = scenario([{ code: `const kept = await refine("Kind words.", "${POLITE}"); const assumed = assume("anything", "${POLITE}"); return kept + " " + assumed;`, finish: true }]);
  assert.equal(await s.run(), 'Kind words. anything');
  assert.equal(s.judged.filter(value => value === 'Kind words.').length, 1);
  assert.ok(s.traces.flatMap(trace => trace.events).some(event => event.kind === 'refinement_assumed'));
});
