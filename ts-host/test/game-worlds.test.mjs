import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { EconomyWorld, CombatWorld, NpcWorld } from '../../applications/game_worlds.mjs';

const paths = {
  economy: fileURLToPath(new URL('../../codebases/game_economy/turn.nl', import.meta.url)),
  combat: fileURLToPath(new URL('../../codebases/game_combat/turn.nl', import.meta.url)),
  npc: fileURLToPath(new URL('../../codebases/game_npc/react.nl', import.meta.url)),
};

async function runPolicy(path, binding, inputs, choice, rootName) {
  const host = new NatlangHost({ host: binding });
  try {
    return await host.run({ source: { kind: 'file', path }, inputs,
      options: { model: { segment_turns: 2 } },
      modelTurn: turn => {
        if (turn.messages.filter(m => m.role === 'assistant').length > 1)
          return { calls: [], text: 'done', completion_tokens: 1 };
        const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes(`function ${rootName}(`)) return { calls: rootName === 'react' ? [
          ['call', { function: 'observe', to: 'let/observation', inputs: {
            actor: 'args/actor', event: 'args/event' } }],
          ['call', { function: 'decide', to: 'let/plan', inputs: {
            observation: 'let/observation' } }],
          ['call', { function: 'apply', to: 'return', inputs: {
            observation: 'let/observation', plan: 'let/plan' } }],
        ] : [
          ['call', { function: 'observe', to: 'let/observation', inputs: {
            actor: 'args/actor' } }],
          ['call', { function: rootName === 'turn' && path === paths.combat ? 'choose' : 'decide',
            to: 'let/plan', inputs: path === paths.combat ?
              { perception: 'let/observation' } : { observation: 'let/observation' } }],
          ['call', { function: 'submit', to: 'return', inputs: path === paths.combat ? {
            actor: 'args/actor', round: 'let/observation/round', plan: 'let/plan',
          } : { actor: 'args/actor', tick: 'let/observation/tick', intent: 'let/plan' } }],
        ], completion_tokens: 1 };
        return { calls: [['write', { path: 'return', value: choice }]], completion_tokens: 1 };
      } });
  } finally { host.close(); }
}

test('natlang merchant policy produces legal trades with conserved money and goods', async () => {
  const merchants = [
    { id: 'alice', cash: 10, goods: { apple: 0, bread: 0 }, offers: {} },
    { id: 'bob', cash: 0, goods: { apple: 2, bread: 0 }, offers: { apple: 3 } },
    { id: 'cara', cash: 0, goods: { apple: 0, bread: 2 }, offers: { bread: 2 } },
  ];
  const world = new EconomyWorld(merchants, { seed: 33 });
  const result = await runPolicy(paths.economy, { economy: world }, { actor: 'alice' },
    { kind: 'buy', seller: 'bob', good: 'apple', quantity: 2 }, 'turn');
  assert.equal(result.outcome.kind, 'done');
  world.submit('bob', 0, { kind: 'pass' }); world.submit('cara', 0, { kind: 'pass' });
  const settled = world.settle();
  assert.equal(settled.outcomes.find(row => row.actor === 'alice').status, 'traded');
  assert.equal(world.observe('alice').cash, 4);
  assert.equal(world.observe('alice').goods.apple, 2);
  assert.equal(world.observe('bob').cash, 6);
  assert.throws(() => world.submit('alice', 0, { kind: 'pass' }), /stale/);
  const other = new EconomyWorld(merchants, { seed: 33 });
  other.submit('cara', 0, { kind: 'pass' }); other.submit('bob', 0, { kind: 'pass' });
  other.submit('alice', 0, { kind: 'buy', seller: 'bob', good: 'apple', quantity: 2 });
  assert.deepEqual(other.settle(), settled);
});

test('natlang combat tactics resolve simultaneous damage and reject stale plans', async () => {
  const world = new CombatWorld([{ id: 'a', x: 1, hp: 5 }, { id: 'b', x: 3, hp: 5 }]);
  const a = await runPolicy(paths.combat, { combat: world }, { actor: 'a' },
    { move: 'right', action: 'attack', target: 'b' }, 'turn');
  const b = await runPolicy(paths.combat, { combat: world }, { actor: 'b' },
    { move: 'left', action: 'attack', target: 'a' }, 'turn');
  assert.equal(a.outcome.kind, 'done'); assert.equal(b.outcome.kind, 'done');
  const round = world.resolve();
  assert.equal(round.fighters.find(row => row.id === 'a').hp, 3);
  assert.equal(round.fighters.find(row => row.id === 'b').hp, 3);
  assert.throws(() => world.submit('a', 0, { move: 'stay', action: 'rest' }), /stale/);
});

test('natlang NPC response uses assigned memory and exact one-time world action', async () => {
  const world = new NpcWorld([
    { id: 'innkeeper', inventory: { key: 1 } },
    { id: 'guard', inventory: { key: 0 } },
  ]);
  const result = await runPolicy(paths.npc, { npc: world },
    { actor: 'innkeeper', event: { from: 'guest', text: 'May I have a key?' } },
    { say: 'Here is a key.', action: 'give', item: 'key', target: 'guest' }, 'react');
  assert.equal(result.outcome.kind, 'done');
  assert.equal(result.value.inventory.key, 0);
  const observation = world.observe('innkeeper', { from: 'guest', text: 'Promise help?' });
  const promised = world.apply(observation, { say: 'I will help.', action: 'promise',
    target: 'guest', detail: 'Find the map' });
  assert.equal(promised.commitments[0].evidence_id, observation.event_id);
  assert.throws(() => world.apply(observation, { say: 'again', action: 'promise',
    target: 'guest', detail: 'Find the map' }), /invalid NPC plan/);
  const forged = world.observe('guard', { from: 'guest', text: 'Hello' });
  assert.throws(() => world.apply({ ...forged, inventory: { key: 99 } },
    { say: 'Hello', action: 'none' }), /invalid NPC plan/);
});
