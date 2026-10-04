import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { test } from 'node:test';
import { createChesstGame, createChesstTacticFixture } from '../dist/self-play/chesst.js';

const rulesPath = process.env.CHESST_RULES_PATH ?? '/home/werg/chesst/chesst-rules.js';
const skip = !existsSync(rulesPath) ? 'Local ChessT rules source is not installed' : false;

test('ChessT adapter loads and pins the external rules source', { skip }, async () => {
  const game = await createChesstGame(rulesPath);
  assert.equal(game.id, 'chesst');
  assert.match(game.revision, /^[a-f0-9]{64}$/);
  assert.equal(game.source.sha256, game.revision);
  assert.equal(game.source.path, rulesPath);
  assert.deepEqual(game.seats, ['gold', 'blue']);
});

test('ChessT adapter exposes an atomic public action and replays it canonically', { skip }, async () => {
  const game = await createChesstGame(rulesPath);
  const state = game.initialize(createChesstTacticFixture(), 17);
  assert.equal(game.actor(state), 'gold');
  const view = game.observe(state, 'gold');
  const action = game.legalActions(state, 'gold').find((candidate) =>
    candidate.type === 'MOVE' && candidate.from?.row === 6 && candidate.from?.col === 3 &&
    candidate.to?.row === 5 && candidate.to?.col === 3);
  assert.ok(action);
  const next = game.apply(state, 'gold', action);
  assert.equal(game.outcome(next), null);
  assert.equal(game.actor(next), 'blue');
  assert.equal(view.decisionPlayer, 'gold');
});

test('ChessT adapter preserves hidden-item noninterference and rejects illegal actions', { skip }, async () => {
  const game = await createChesstGame(rulesPath);
  const dagger = createChesstTacticFixture();
  const tome = createChesstTacticFixture();
  dagger.board[1][0].item.id = 'dagger';
  tome.board[1][0].item.id = 'tome';
  const daggerState = game.initialize(dagger, 0);
  const tomeState = game.initialize(tome, 0);
  assert.deepEqual(game.observe(daggerState, 'gold'), game.observe(tomeState, 'gold'));
  assert.throws(() => game.apply(daggerState, 'gold', { type: 'TELEPORT', target: { row: 4, col: 4 } }), /Illegal ChessT action/);
  assert.throws(() => game.apply(daggerState, 'blue', {}), /wrong decision seat/);
});

test('ChessT adapter handles item subphases and canonical terminal outcomes', { skip }, async () => {
  const game = await createChesstGame(rulesPath);
  let state = game.initialize(createChesstTacticFixture(), 1);
  state = game.apply(state, 'gold', { type: 'BEGIN_ITEM', from: { row: 6, col: 3 }, item: 'swiftness' });
  assert.equal(state.phase, 'ITEM_TARGET');
  state = game.apply(state, 'gold', { type: 'ITEM_TARGET', target: { row: 6, col: 3 } });
  assert.equal(state.phase, 'FORCED_ITEM_MOVE');
  assert.ok(game.legalActions(state, 'gold').some((action) => action.type === 'MOVE'));

  const capture = createChesstTacticFixture();
  capture.board[6][3] = null;
  capture.board[1][4] = { type: 'rook', color: 'gold', hasMoved: false };
  const winning = game.initialize(capture, 2);
  const action = game.legalActions(winning, 'gold').find((candidate) =>
    candidate.type === 'MOVE' && candidate.to?.row === 0 && candidate.to?.col === 4);
  assert.ok(action);
  const terminal = game.apply(winning, 'gold', action);
  assert.deepEqual(game.outcome(terminal), {
    scores: { gold: 1, blue: 0 },
    reason: 'royalransom:king-captured',
  });
});
