import assert from 'node:assert/strict';
import { test } from 'node:test';
import { formatType } from '../dist/native/types.js';
import { session as open, ts } from './support/natlang.mjs';

test('declaring a variable with a function\'s name says to keep the result under another name', async () => {
  const { session } = open({ type: '(ids: string[]) => number', instructions: 'Count.', args: { ids: ['a'] },
    codebase: { plan: ts('plan', 'export default function plan(id: string): string { return id; }') } });
  const outcome = await session.applyAsync('eval', { code: 'const plan = plan(ids[0]);\nplan' });
  assert.notEqual(outcome.kind, 'ok');
  assert.match(outcome.text, /plan is already a function in this scope; call plan\(\.\.\.\) directly and keep its result in a variable with another name, such as planResult\./);
});

test('a mapped callback that wraps the call keeps the elements\' own type, not the callee return type', async () => {
  const { lam, session } = open({ type: '(ids: string[]) => number', instructions: 'Count.', args: { ids: ['a', 'b'] },
    codebase: {
      choose: ts('choose', 'export default async function choose(id: string): Promise<string> { return id + "!"; }'),
    } });
  const wrapped = await session.applyAsync('eval', { code:
    'const actor = "x";\nconst picks = await Promise.all(ids.map(async id => ({ actor, intent: await choose(id) })));\npicks' });
  assert.equal(wrapped.kind, 'ok', wrapped.text);
  assert.deepEqual(wrapped.value, [{ actor: 'x', intent: 'a!' }, { actor: 'x', intent: 'b!' }]);
  assert.notEqual(formatType(lam.letTypes.picks), 'string[]');
  const direct = await session.applyAsync('eval', { code: 'const names = await Promise.all(ids.map(async id => await choose(id)));\nnames' });
  assert.equal(direct.kind, 'ok', direct.text);
  assert.equal(formatType(lam.letTypes.names), 'string[]');
});
