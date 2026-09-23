import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import { EconomyWorld, CombatWorld, NpcWorld, tradeTurn, combatTurn, npcReact } from '../../applications/dist/games/index.js';
import { scriptedModel } from './support/natlang.mjs';

/** Run one policy call with a model that answers with `choice`. */
function policy(choice) {
  const model = scriptedModel(() => `result = ${JSON.stringify(choice)}`);
  return { runtime: createNatlangRuntime({ model: model.driver }), model };
}

test('a natlang merchant policy produces legal trades with conserved money and goods', async () => {
  const merchants = [
    { id: 'alice', cash: 10, goods: { apple: 0, bread: 0 }, offers: {} },
    { id: 'bob', cash: 0, goods: { apple: 2, bread: 0 }, offers: { apple: 3 } },
    { id: 'cara', cash: 0, goods: { apple: 0, bread: 2 }, offers: { bread: 2 } },
  ];
  const world = new EconomyWorld(merchants, { seed: 33 });
  const { runtime, model } = policy({ kind: 'buy', seller: 'bob', good: 'apple', quantity: 2 });
  assert.equal((await runtime.run(() => tradeTurn(world, 'alice'))).status, 'accepted');
  assert.match(model.openings[0], /observation: MarketObservation|observation: \{/);
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
  await Promise.all([
    policy({ move: 'right', action: 'attack', target: 'b' }).runtime.run(() => combatTurn(world, 'a')),
    policy({ move: 'left', action: 'attack', target: 'a' }).runtime.run(() => combatTurn(world, 'b'))]);
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
  const result = await policy({ say: 'Here is a key.', action: 'give', item: 'key', target: 'guest' }).runtime
    .run(() => npcReact(world, 'innkeeper', { from: 'guest', text: 'May I have a key?' }));
  assert.equal(result.inventory.key, 0);
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
