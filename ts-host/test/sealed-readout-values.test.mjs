// A call answered by a decision readout offers no tools, so it cannot read its eval scope: its opening shows each value
// whole within the context budget, and a cut's note never names a variable it cannot reach (DECISIONS 2026-11-25
// "Judges never decide on a cut argument"). NATLANG_READOUT_WHOLE_VALUES=0 restores the scope listing's cut.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, decide, loadVirtualNatlang } from '../dist/index.js';
import { sealedBudgets } from '../dist/native/agent.js';
import { sealedValuesChars, SCOPE_VALUE_CHARS } from '../dist/native/cutoff.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';

const judge = readout => `---\nargs: { text: string }\nreturns: boolean\n${readout ? `readout: ${readout}\n` : ''}---\nDecide whether text is polite.\n`;

/** The text of every request the model was sent (readout scoring and generation both). */
async function opening(value, { readout = 'instant', contextTokens = 16384, args = 'text' } = {}) {
  const requests = [];
  const driver = Object.assign(async request => {
    requests.push(request);
    return { calls: [['return_result', { status: 'success', value: true }]], completion_tokens: 1 };
  }, { decide: async request => { requests.push(request); return { log_probs: [Math.log(0.7), Math.log(0.3)], tokens: [1, 1] }; } });
  const runtime = createNatlangRuntime({ calls: false, model: { driver, contextTokens }, seed: { mode: 'backend' } });
  const fn = loadVirtualNatlang({ 'judge.nl': args === 'text' ? judge(readout) : args }, 'judge.nl');
  await runtime.run(() => decide(fn, value));
  return JSON.stringify(requests[0].messages);
}

test('a readout shows a long argument whole within the context budget, with no note naming its variable', async () => {
  const value = 'please and thank you. '.repeat(1000);  // 22,000 characters: past the scope listing's 4,096
  const shown = await opening(value);
  assert.ok(shown.includes(JSON.stringify(JSON.stringify(value)).slice(1, -1)), 'the whole value is in the opening');
  assert.ok(!shown.includes('cut off'));
});

test('past the context budget the readout cuts, and its note does not point at a scope it cannot read', async () => {
  const value = 'x'.repeat(200000);
  const shown = await opening(value);
  assert.match(shown, /<<cut off: \d+ of 200000 characters not shown>>/);
  assert.ok(!/holds the complete value|holds all of it/.test(shown));
  assert.ok(shown.length > sealedValuesChars(16384) * 0.9, 'the budget, not the scope listing page, bounds it');
});

test('a tool-loop call keeps the scope listing: cut at its page and the variable named', async () => {
  const value = 'please and thank you. '.repeat(1000);
  const shown = await opening(value, { readout: null });
  assert.match(shown, /only this display is shortened, text holds the complete value/);
});

test('NATLANG_READOUT_WHOLE_VALUES=0 restores the scope listing in a readout', async () => {
  process.env.NATLANG_READOUT_WHOLE_VALUES = '0';
  try {
    const shown = await opening('please and thank you. '.repeat(1000));
    assert.match(shown, /only this display is shortened, text holds the complete value/);
  } finally { delete process.env.NATLANG_READOUT_WHOLE_VALUES; }
});

test('a record argument of a readout shows every field the budget holds, not the listing\'s first 24', async () => {
  const record = Object.fromEntries(Array.from({ length: 40 }, (_, index) => [`field${index}`, index]));
  const source = '---\nargs: { row: unknown }\nreturns: boolean\nreadout: instant\n---\nDecide whether row is complete.\n';
  const shown = await opening(record, { args: source });
  assert.ok(shown.includes('field39: 39') && !shown.includes('cut off'));
});

test('sealedBudgets: values that fit an even share are whole, the rest share what is left (never below the listing\'s)', () => {
  const render = (value, budget) => value.length <= budget ? value : value.slice(0, budget) + '<<cut off: …>>';
  const budgets = sealedBudgets([['arg:a', 'a'.repeat(100)], ['arg:b', 'b'.repeat(50000)], ['arg:c', 'c'.repeat(50000)]], 20000, render);
  assert.equal(budgets.get('arg:a'), 20000);
  assert.equal(budgets.get('arg:b'), Math.max(SCOPE_VALUE_CHARS, Math.floor((20000 - 100) / 2)));
  assert.equal(sealedBudgets([['arg:a', 'a']], Infinity, render).get('arg:a'), Infinity);
  assert.equal(sealedValuesChars(null), Infinity);
});

test('under learned compaction a readout\'s cut argument gets the compact-params state, then its text, with no holder', async () => {
  const ID = 'nzs1_' + 'a'.repeat(52);
  const operatorCalls = [], requests = [];
  const capture = async () => ({ id: ID, dialect: 'native-state/test', type: 'Neuralese<string>', length: 3, width: 8,
    producer: { native_source_capture: { segments: [], payload_start: 5, payload_stop: 8 } } });
  const driver = Object.assign(async request => {
    if (request.invocation_id?.includes('#operator-')) {
      operatorCalls.push(request);
      return { text: 'a polite text', completion_tokens: 2, raw_response: { x_natlang_rl_capture: {
        prompt: [{ kind: 'tokens', token_ids: [1, 2] }], generated_inputs: [{ token_id: 9, input_role: 'ordinary' }] } } };
    }
    requests.push(request);
    return { text: 'true', calls: [], completion_tokens: 1 };
  }, { neuralese: true, decide: async request => { requests.push(request); return { log_probs: [-0.1, -2], tokens: [1, 1] }; } });
  const runtime = createNatlangRuntime({ calls: false, model: { driver, contextTokens: 16384 }, seed: { mode: 'backend' },
    neuralese: { store: new MemoryNeuraleseStore(), dialect: 'native-state/test', capture, compaction: 'nl' } });
  const fn = loadVirtualNatlang({ 'judge.nl': judge('instant') }, 'judge.nl');
  await runtime.run(() => decide(fn, 'x'.repeat(200000))).catch(() => undefined);
  assert.equal(operatorCalls.length, 1, 'the value past the budget is compacted');
  assert.match(operatorCalls[0].messages[1].content, /answers: Decide whether text is polite\./);
  // A short value is shown whole and never compacted.
  operatorCalls.length = 0;
  await runtime.run(() => decide(fn, 'please and thank you. '.repeat(1000))).catch(() => undefined);
  assert.equal(operatorCalls.length, 0);
});

test('operatorSite mirrors a sealed compact-params site: no holder, the state followed by the cut text', async () => {
  const { operatorSite } = await import('../dist/native/operator-site.js');
  const site = operatorSite({ operator: 'compact-params', value: 'x'.repeat(200000), purpose: 'p', argument: { name: 'text', type: 'string' },
    context_tokens: 16384, sealed: true });
  assert.equal(site.compacted, true);
  assert.ok(!/holds/.test(site.text) && !/holds/.test(site.state.before + site.state.after));
  assert.match(site.state.after, /^ {2}\/\/ view of the whole value; its text follows as far as it fits\n"x+" <<cut off:/);
  const whole = operatorSite({ operator: 'compact-params', value: 'y'.repeat(20000), purpose: 'p', argument: { name: 'text', type: 'string' },
    context_tokens: 16384, sealed: true });
  assert.equal(whole.compacted, false);
});
