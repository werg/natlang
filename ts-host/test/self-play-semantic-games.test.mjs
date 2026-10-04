import assert from 'node:assert/strict';
import test from 'node:test';
import { semanticGames, semanticScenarioCases } from '../dist/self-play/semantic-games.js';
import { IllegalGameAction } from '../dist/self-play/types.js';

const games = new Map(semanticGames.map(game => [game.id, game]));
const cases = semanticScenarioCases();
const forFamily = family => cases.filter(item => item.family === family);
const json = value => JSON.stringify(value);
const bounded = outcome => Object.values(outcome.scores).every(score => Number.isFinite(score) && score >= 0 && score <= 1);

test('semantic scenario catalog has stable source groups and at least four original cases per game', () => {
  assert.equal(new Set(cases.map(item => item.group)).size, cases.length);
  for (const game of semanticGames) assert.ok(forFamily(game.id).length >= 4, `${game.id} needs at least four scenarios`);
  assert.ok(cases.every(item => item.group.startsWith(`semantic-game/${item.family}/`)));
});

test('clue-intercept hides codebook and current answer from interceptor, then reveals prior answers', () => {
  const game = games.get('clue-intercept'), scenario = forFamily(game.id)[0].scenario;
  let state = game.initialize(scenario, 29);
  const original = json(state);
  const sender = game.observe(state, 'sender'), receiver = game.observe(state, 'receiver'), interceptor = game.observe(state, 'interceptor');
  assert.deepEqual(sender.codebook, receiver.codebook);
  const otherSeed = game.initialize(scenario, 30);
  assert.notDeepEqual(state.slots, otherSeed.slots, 'the private codebook mapping and opaque slot ids vary by seed');
  assert.notDeepEqual(state.rounds[0].cards.map(card => card.id), otherSeed.rounds[0].cards.map(card => card.id));
  assert.equal(interceptor.codebook, undefined);
  assert.equal(interceptor.target, undefined);
  assert.ok(interceptor.slots.length === 4);
  assert.ok(sender.clueCards.every(card => !Object.hasOwn(card, 'target')));
  assert.equal(json(state), original, 'observe and later transitions must leave inputs unchanged');

  const round = scenario.rounds[0], card = state.rounds[0].cards.find(item => item.target === round.target);
  const targetSlot = state.slots.find(slot => slot.word === round.target).id;
  state = game.apply(state, 'sender', { type: 'clue', cardId: card.id });
  assert.equal(game.actor(state), 'receiver');
  const receiverView = game.observe(state, 'receiver');
  assert.equal(receiverView.clue, card.text);
  assert.equal(receiverView.target, undefined);
  state = game.apply(state, 'receiver', { type: 'guess', slotId: targetSlot });
  const interceptorView = game.observe(state, 'interceptor');
  assert.equal(interceptorView.receiverGuess, undefined);
  assert.equal(interceptorView.codebook, undefined);
  const wrongSlot = state.slots.find(slot => slot.id !== targetSlot).id;
  state = game.apply(state, 'interceptor', { type: 'guess', slotId: wrongSlot });
  assert.equal(state.history[0].answer, targetSlot);
  assert.equal(game.actor(state), 'sender');
  assert.deepEqual(game.observe(state, 'interceptor').history[0], state.history[0]);
});

test('clue-intercept scripted semantic policy wins; irrelevant clues deterministically score zero', () => {
  const game = games.get('clue-intercept');
  for (const row of forFamily(game.id)) {
    let state = game.initialize(row.scenario, 11);
    while (!game.outcome(state)) {
      const seat = game.actor(state), round = row.scenario.rounds[state.round];
      if (seat === 'sender') {
        const card = state.rounds[state.round].cards.find(item => item.target === round.target);
        state = game.apply(state, seat, { type: 'clue', cardId: card.id });
      } else if (seat === 'receiver') {
        state = game.apply(state, seat, { type: 'guess', slotId: state.slots.find(slot => slot.word === round.target).id });
      } else {
        const target = state.slots.find(slot => slot.word === round.target).id;
        state = game.apply(state, seat, { type: 'guess', slotId: state.slots.find(slot => slot.id !== target).id });
      }
    }
    assert.equal(game.outcome(state).scores.sender, 1, row.group);
    assert.equal(game.outcome(state).scores.interceptor, 0, row.group);
    assert.ok(bounded(game.outcome(state)));
  }
  const row = forFamily(game.id)[0], scenario = row.scenario;
  let state = game.initialize(scenario, 4), round = scenario.rounds[0], current = state.rounds[0];
  const irrelevant = current.cards.find(item => item.target !== round.target);
  state = game.apply(state, 'sender', { type: 'clue', cardId: irrelevant.id });
  state = game.apply(state, 'receiver', { type: 'guess', slotId: state.slots.find(slot => slot.word === round.target).id });
  const target = state.slots.find(slot => slot.word === round.target).id;
  state = game.apply(state, 'interceptor', { type: 'guess', slotId: state.slots.find(slot => slot.id !== target).id });
  assert.equal(state.roundTeams[0], 0);
  round = scenario.rounds[1]; current = state.rounds[1];
  const relevant = current.cards.find(item => item.target === round.target);
  state = game.apply(state, 'sender', { type: 'clue', cardId: relevant.id });
  const nextTarget = state.slots.find(slot => slot.word === round.target).id;
  state = game.apply(state, 'receiver', { type: 'guess', slotId: nextTarget });
  state = game.apply(state, 'interceptor', { type: 'guess', slotId: state.slots.find(slot => slot.id !== nextTarget).id });
  assert.equal(game.outcome(state).scores.sender, 0.5);
});

test('clue-intercept rejects illegal actor and card actions without mutating the state', () => {
  const game = games.get('clue-intercept'), state = game.initialize(forFamily(game.id)[0].scenario, 2), before = json(state);
  assert.throws(() => game.apply(state, 'receiver', { type: 'clue', cardId: 'bad' }), IllegalGameAction);
  assert.throws(() => game.apply(state, 'sender', { type: 'clue', cardId: 'missing' }), IllegalGameAction);
  assert.equal(json(state), before);
});

test('evidence-bluff keeps truth host-private until terminal and scores exact reviewer judgments', () => {
  const game = games.get('evidence-bluff');
  for (const row of forFamily(game.id)) {
    let state = game.initialize(row.scenario, 0);
    const advocate = game.observe(state, 'advocate'), reviewerAtStart = game.observe(state, 'reviewer');
    assert.equal(advocate.privateVerdict, row.scenario.verdict);
    assert.equal(advocate.claim, row.scenario.claim);
    assert.equal(Object.hasOwn(advocate, 'rationale'), false);
    assert.equal(reviewerAtStart.verdict, undefined);
    assert.equal(reviewerAtStart.dossier, undefined);
    assert.equal(reviewerAtStart.claim, undefined);
    const card = row.scenario.arguments[0];
    state = game.apply(state, 'advocate', { type: 'argument', argumentId: card.id });
    const reviewer = game.observe(state, 'reviewer');
    assert.equal(reviewer.verdict, undefined);
    assert.equal(reviewer.claim, row.scenario.claim);
    assert.equal(Object.hasOwn(reviewer, 'rationale'), false);
    assert.equal(reviewer.argument, card.text);
    assert.equal(Object.hasOwn(reviewer, 'verdict'), false);
    state = game.apply(state, 'reviewer', { type: 'verdict', value: row.scenario.verdict });
    assert.equal(game.outcome(state).scores.reviewer, 1);
    assert.equal(game.outcome(state).scores.advocate, 0);
    assert.ok(bounded(game.outcome(state)));
  }
});

test('evidence-bluff rejects invalid guesses and supports a successful bluff', () => {
  const game = games.get('evidence-bluff'), row = forFamily(game.id).find(item => item.scenario.verdict === false);
  let state = game.initialize(row.scenario, 7);
  assert.throws(() => game.apply(state, 'reviewer', { type: 'verdict', value: false }), IllegalGameAction);
  state = game.apply(state, 'advocate', { type: 'argument', argumentId: row.scenario.arguments[0].id });
  assert.throws(() => game.apply(state, 'reviewer', { type: 'verdict', value: 'false' }), IllegalGameAction);
  state = game.apply(state, 'reviewer', { type: 'verdict', value: true });
  assert.equal(game.outcome(state).scores.advocate, 1);
  assert.equal(game.outcome(state).scores.reviewer, 0);
});

test('meaning-bargain reveals only each negotiator’s own preference and accepts a mutually useful offer', () => {
  const game = games.get('meaning-bargain'), row = forFamily(game.id)[0];
  let state = game.initialize(row.scenario, 3);
  const proposer = game.observe(state, 'proposer'), responder = game.observe(state, 'responder');
  assert.equal(proposer.preference, row.scenario.preferences.proposer);
  assert.equal(responder.preference, row.scenario.preferences.responder);
  assert.equal(json(proposer).includes(JSON.stringify(row.scenario.utilities)), false);
  assert.equal(json(responder).includes(JSON.stringify(row.scenario.utilities)), false);
  assert.equal(proposer.opponentPreference, undefined);
  assert.equal(responder.opponentPreference, undefined);
  const plan = row.scenario.plans[0];
  state = game.apply(state, 'proposer', { type: 'offer', planId: plan.id, message: 'Does this fit both schedules?' });
  assert.equal(game.actor(state), 'responder');
  assert.equal(game.legalActions(state, 'responder')[0].type, 'accept');
  assert.equal(game.legalActions(state, 'proposer').length, 0);
  assert.equal(game.observe(state, 'responder').currentOffer.plan, plan.text);
  state = game.apply(state, 'responder', { type: 'accept' });
  assert.deepEqual(game.outcome(state).scores, { proposer: row.scenario.utilities.proposer[plan.id], responder: row.scenario.utilities.responder[plan.id] });
  assert.ok(bounded(game.outcome(state)));
});

test('meaning-bargain decline and fixed round exhaustion are legitimate zero-score endings', () => {
  const game = games.get('meaning-bargain'), scenario = forFamily(game.id)[0].scenario;
  let state = game.initialize(scenario, 9);
  state = game.apply(state, 'proposer', { type: 'decline' });
  assert.deepEqual(game.outcome(state).scores, { proposer: 0, responder: 0 });
  state = game.initialize(scenario, 9);
  for (let round = 0; round < 6 && !game.outcome(state); round++) {
    const seat = game.actor(state), planId = scenario.plans[round % scenario.plans.length].id;
    state = game.apply(state, seat, { type: 'offer', planId });
  }
  assert.equal(game.outcome(state), null, 'the recipient gets a chance to accept or decline the sixth offer');
  assert.deepEqual(game.legalActions(state, game.actor(state)).map(action => action.type), ['accept', 'decline']);
  state = game.apply(state, game.actor(state), { type: 'decline' });
  assert.equal(game.outcome(state).reason, 'round limit reached without agreement');
  assert.deepEqual(game.outcome(state).scores, { proposer: 0, responder: 0 });
});

test('meaning-bargain supports a counteroffer followed by acceptance', () => {
  const game = games.get('meaning-bargain'), scenario = forFamily(game.id)[0].scenario;
  let state = game.initialize(scenario, 5);
  state = game.apply(state, 'proposer', { type: 'offer', planId: scenario.plans[0].id });
  state = game.apply(state, 'responder', { type: 'offer', planId: scenario.plans[1].id, message: 'This alternative works better.' });
  assert.equal(game.actor(state), 'proposer');
  assert.equal(game.observe(state, 'proposer').currentOffer.planId, scenario.plans[1].id);
  state = game.apply(state, 'proposer', { type: 'accept' });
  assert.equal(game.outcome(state).scores.proposer, scenario.utilities.proposer[scenario.plans[1].id]);
  assert.equal(game.outcome(state).scores.responder, scenario.utilities.responder[scenario.plans[1].id]);
});

test('meaning-bargain rejects offers outside the finite menu and wrong actors', () => {
  const game = games.get('meaning-bargain'), state = game.initialize(forFamily(game.id)[0].scenario, 1), before = json(state);
  assert.throws(() => game.apply(state, 'responder', { type: 'decline' }), IllegalGameAction);
  assert.throws(() => game.apply(state, 'proposer', { type: 'offer', planId: 'invented' }), IllegalGameAction);
  assert.equal(json(state), before);
});

test('all game transitions are deterministic immutable JSON and terminal scores stay bounded', () => {
  for (const game of semanticGames) for (const row of forFamily(game.id)) {
    const first = game.initialize(row.scenario, 123), second = game.initialize(row.scenario, 123);
    assert.deepEqual(first, second, `${row.group} initialization`);
    assert.doesNotThrow(() => JSON.stringify(first));
    const before = json(first), seat = game.actor(first), action = game.legalActions(first, seat)[0];
    const nextA = game.apply(first, seat, action), nextB = game.apply(second, seat, action);
    assert.deepEqual(nextA, nextB, `${row.group} transition`);
    assert.equal(json(first), before, `${row.group} transition mutated input`);
    assert.doesNotThrow(() => JSON.stringify(nextA));
  }
});

test('negotiation includes public experience without exposing the other preference or utility', () => {
  const game=semanticGames.find(g=>g.id==='meaning-bargain');
  const scenario=semanticScenarioCases().find(row=>row.family===game.id).scenario;
  let state=game.initialize(scenario,17);
  const first=game.legalActions(state,'proposer').find(action=>action.type==='offer');
  state=game.apply(state,'proposer',{...first,message:'I need a quiet place.'});
  const response=game.legalActions(state,'responder').find(action=>action.type==='offer'&&action.planId!==first.planId);
  state=game.apply(state,'responder',response);
  const view=game.observe(state,'proposer');
  assert.equal(view.history.length,2);assert.equal(view.history[0].message,'I need a quiet place.');
  assert.equal(view.history[1].planId,response.planId);
  assert.ok(!JSON.stringify(view).includes(scenario.preferences.responder));
  assert.equal(view.utilities,undefined);
});
