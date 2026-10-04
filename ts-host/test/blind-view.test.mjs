/** Reward-blind views (LEARNING_CONTINUUM.md §10.1): schema stripping per visibility class and the gate. */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { blindView, checkBlindView } from '../dist/improvement/blind-view.js';

const events = [
  { kind: 'invocation', seq: 1, call_id: 'c1', definition: 'solve', inputs: { q: 'list overdue invoices' }, captures: { secret: 1 } },
  { kind: 'model_request', seq: 2, call_id: 'c1', turn: 0, prompt_tokens: 900, messages: [{ role: 'user', content: 'list overdue invoices' },
    { role: 'tool', content: 'rows: 3' }] },
  { kind: 'proposal', seq: 3, call_id: 'c1', turn: 0, text: 'I will query.', calls: [{ name: 'sql', arguments: '{"q":"select"}' }] },
  { kind: 'action', seq: 4, call_id: 'c1', name: 'sql', arguments: { q: 'select' }, result_text: 'rows: 3', outcome: 'ok',
    diagnostics: [], surface: 'x' },
  { kind: 'effect', seq: 5, call_id: 'c1', service: 'db', method: 'query', args: {}, result: { rows: 3, score: 0.4 } },
  { kind: 'state', seq: 6, outcome: 'success', value: ['inv-7'] },
  { kind: 'evaluation', seq: 7, quality: 0.5, passed: false, expected: ['inv-7', 'inv-9'] },
  { kind: 'skill_use', seq: 8, skill_name: 'billing' },
];
const attempt = { inputs: { q: 'list overdue invoices' }, traces: [{ events }] };

test('observations keep environment responses; strict withholds them; feedback never appears', () => {
  const observed = blindView({ visibility: 'reward-blind-observations', function: { 'solve.nl': 'List the overdue invoices.' }, attempts: [attempt] });
  assert.deepEqual(observed.attempts[0].events.map(e => e.kind), ['invocation', 'model_request', 'proposal', 'action', 'effect', 'state']);
  const action = observed.attempts[0].events.find(e => e.kind === 'action');
  assert.equal(action.result_text, 'rows: 3');
  assert.equal(action.surface, undefined);
  assert.deepEqual(observed.attempts[0].events.find(e => e.kind === 'effect').result, { rows: 3 }, 'nested score stripped');
  assert.equal(observed.attempts[0].events[0].captures, undefined);
  assert.deepEqual(checkBlindView(observed), []);

  const strict = blindView({ visibility: 'reward-blind-strict', function: { 'solve.nl': 'List the overdue invoices.' }, attempts: [attempt] });
  const strictAction = strict.attempts[0].events.find(e => e.kind === 'action');
  assert.equal(strictAction.result_text, undefined);
  assert.equal(strict.attempts[0].events.find(e => e.kind === 'effect').result, undefined);
  assert.equal(strict.attempts[0].events.find(e => e.kind === 'model_request').messages[1].content, '[environment response withheld]');
  assert.deepEqual(strict.attempts[0].events.find(e => e.kind === 'state').value, ['inv-7'], 'the function\'s own output stays');
  assert.deepEqual(checkBlindView(strict), []);
});

test('the gate finds forbidden keys, unknown kinds, environment fields in strict views and sealed fragments', () => {
  const view = blindView({ visibility: 'reward-blind-strict', function: { 'solve.nl': 'x' }, attempts: [attempt] });
  const tampered = { ...view, attempts: [{ ...view.attempts[0], events: [...view.attempts[0].events,
    { kind: 'evaluation', seq: 9 }, { kind: 'action', seq: 10, name: 'sql', result_text: 'rows: 3' }, { kind: 'state', seq: 11, value: { quality: 1 } }] }] };
  const problems = checkBlindView(tampered, ['inv-9 is overdue since March']);
  assert.ok(problems.some(p => p.includes('event kind evaluation')));
  assert.ok(problems.some(p => p.includes('environment field result_text')));
  assert.ok(problems.some(p => p.includes('forbidden key quality')));
  assert.deepEqual(checkBlindView(view, ['inv-7']).length, 1, 'a sealed answer fragment in the function output is caught');
});
