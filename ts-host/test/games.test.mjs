import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';
import { createNatlangRuntime } from '../dist/index.js';
import { readFileSync } from 'node:fs';
import { Session, createArena, createEconomy, createVillage, crispSettings, defaultSettings, playTurn }
  from '../../applications/dist/games/index.js';
import { appCrisp, scriptedModel, withJudge } from './support/natlang.mjs';

const CRISP = await appCrisp('games');

// ---------------------------------------------------------------- the crisp checks, loaded straight from their sources

/** A callable-folder TypeScript module (commit, order, reference), bundled so its checks can be called directly. */
async function crisp(path) {
  const entry = fileURLToPath(new URL(`../../applications/games/games/${path}`, import.meta.url));
  const out = await build({ entryPoints: [entry], bundle: true, format: 'esm', write: false, platform: 'node', logLevel: 'silent' });
  return import(`data:text/javascript;base64,${Buffer.from(out.outputFiles[0].text).toString('base64')}`);
}
const economyCommit = (await crisp('economy/commit.ts')).default;
const economyOrder = await crisp('economy/order.ts');
const economyReference = (await crisp('economy/settle/reference.ts')).default;
const combatCommit = (await crisp('combat/commit.ts')).default;
const combatReference = (await crisp('combat/resolve/reference.ts')).default;
const npcCommit = (await crisp('npc/commit.ts')).default;

// ---------------------------------------------------------------- the stages, scripted

/** What each merchant chooses. */
const TRADES = {
  alice: { kind: 'buy', seller: 'bob', good: 'apple', quantity: 2 },
  bob: { kind: 'pass' },
  cara: { kind: 'pass' },
};
const STAGES = [
  ['Play one turn of scene', 'games'], ['Play one tick of the economy', 'economy'], ['Play one round of the arena', 'combat'],
  ['Play the reaction of the NPC', 'npc'], ['Choose the intent of the merchant', 'choose'], ['Judge submission.intent', 'judge'],
  ['Settle the intents in ordered', 'settle'], ['Decide submission.intent for the merchant', 'quote'],
  ['Choose the tactic of the fighter', 'tactic'], ['Judge submission.plan, the tactic', 'tacticJudge'],
  ['Resolve the round with the stages', 'resolve'], ['Move the fighters', 'move'], ['Find the hits', 'strike'],
  ['Wound the fighters', 'wound'], ['Advance the cooldown timers', 'recover'],
  ['Write the notes that the NPC', 'remember'], ['Respond for the NPC', 'respond'], ['Judge plan, the plan of the NPC', 'planJudge'],
  ['Tell the turn of a world', 'story'],
];

const SCRIPTS = {
  games: `
    let turn;
    if (scene.kind === 'economy') turn = await economy(scene.state, settings);
    else if (scene.kind === 'combat') turn = await combat(scene.state, settings);
    else turn = await npc(scene.state, scene.actor, scene.event, settings);
    const narration = turn.ok ? await narrate(scene.kind, turn.events, settings) : 'The turn did not happen: ' + turn.problem;
    return { kind: scene.kind, ok: turn.ok, state: turn.state, events: turn.events, log: turn.log, problem: turn.problem, narration };`,
  economy: `
    const ids = state.merchants.map(m => m.id), log = [];
    const picks: Submission[] = await Promise.all(ids.map(async id => ({ actor: id, intent: await choose(await observe(state, id)) })));
    const verdicts = await Promise.all(picks.map(s => validate(ids, s, settings)));
    const checked: Submission[] = picks.map((s, i) => verdicts[i].ok ? s : (log.push(s.actor + ': ' + verdicts[i].reason), { actor: s.actor, intent: { kind: 'pass' } }));
    const sequence = await order(state.seed, state.tick, checked.map(s => s.actor));
    const ordered = sequence.map(a => checked.find(s => s.actor === a));
    let problem = '', committed;
    for (const attempt of [0, 1]) {
      const settlement = await settle(state, ordered, settings, problem);
      committed = await commit(state, { basis: state.tick, order: sequence, submissions: ordered, settlement });
      if (committed.ok) break;
      problem = committed.problem;
    }
    if (!committed.ok) return { ok: false, state, events: [], log, problem: committed.problem };
    return { ok: true, state: committed.state, events: committed.events, log, problem: '' };`,
  combat: `
    const log = [];
    const alive = state.fighters.filter(f => f.hp > 0);
    const submissions: CombatSubmission[] = await Promise.all(alive.map(async f => ({ actor: f.id, plan: await tactic(await observe(state, f.id)) })));
    const verdicts = await Promise.all(submissions.map(s => validate(state, s, settings)));
    const checked: CombatSubmission[] = submissions.map((s, i) => verdicts[i].ok ? s : (log.push(s.actor + ': ' + verdicts[i].reason), { actor: s.actor, plan: { move: 'stay', action: 'rest' } }));
    let problem = '', committed;
    for (const attempt of [0, 1]) {
      const resolution = await resolve(state, checked, settings, problem);
      committed = await commit(state, { basis: state.round, submissions: checked, resolution });
      if (committed.ok) break;
      problem = committed.problem;
    }
    if (!committed.ok) return { ok: false, state, events: [], log, problem: committed.problem };
    return { ok: true, state: committed.state, events: committed.events, log, problem: '' };`,
  npc: `
    const log = [];
    const observation = await observe(state, actor, event);
    const notes = await remember(observation, settings);
    let chosen = await respond(observation, notes);
    const verdict = await validate(observation, chosen, settings);
    if (!verdict.ok) { log.push(actor + ': ' + verdict.reason); chosen = { say: chosen.say, action: 'none' }; }
    const committed = await commit(state, { basis: state.version, actor, event_id: observation.event_id, event, notes, plan: chosen });
    if (!committed.ok) return { ok: false, state, events: [], log, problem: committed.problem };
    return { ok: true, state: committed.state, events: committed.events, log, problem: '' };`,
  judge: `
    const i = submission.intent;
    const ok = i.kind === 'pass' || (known.includes(i.seller) && i.seller !== submission.actor && Number.isSafeInteger(i.quantity) && i.quantity >= 1);
    return { actor: submission.actor, ok, reason: ok ? 'a well-formed buy' : 'a buy names another merchant of this economy as seller' };`,
  settle: `
    let running = merchants; const outcomes = [], entries = [];
    for (const s of ordered) {
      const outcome = await quote(running, s);
      const made = await ledger.entriesOf(outcome);
      running = await ledger.post(running, made);
      outcomes.push(outcome); entries.push(...made);
    }
    return { outcomes, entries };`,
  quote: `
    const { actor, intent } = submission;
    if (intent.kind === 'pass') return { actor, status: 'pass' };
    const buyer = running.find(m => m.id === actor), seller = running.find(m => m.id === intent.seller);
    const reject = reason => ({ actor, status: 'rejected', reason });
    if (!seller || seller.id === actor) return reject('seller is not another merchant');
    const price = seller.offers[intent.good];
    if (!Number.isSafeInteger(price)) return reject('seller does not offer the good');
    const total = intent.quantity * price;
    if (!Number.isSafeInteger(total)) return reject('total is not a safe integer');
    if ((seller.goods[intent.good] ?? 0) < intent.quantity) return reject('seller lacks the stock');
    if (buyer.cash < total) return reject('buyer lacks the cash');
    return { actor, status: 'traded', seller: seller.id, good: intent.good, quantity: intent.quantity, total };`,
  tacticJudge: `
    const self = state.fighters.find(f => f.id === submission.actor), p = submission.plan;
    const target = state.fighters.find(f => f.id === p.target);
    const ok = !!self && self.hp > 0 && ['left', 'stay', 'right'].includes(p.move) && ['attack', 'guard', 'rest'].includes(p.action) &&
      (p.action !== 'attack' || (!!target && target.id !== submission.actor && target.hp > 0));
    return { actor: submission.actor, ok, reason: ok ? 'a legal tactic' : 'an attack names another living fighter as target' };`,
  resolve: `
    const moved = await move(fighters, submissions, width);
    const hits = await strike(moved, submissions);
    const damage = {};
    for (const h of hits) damage[h.target] = (damage[h.target] ?? 0) + h.amount;
    const wounded = await wound(moved, hits);
    return { fighters: await recover(wounded, hits), hits, damage };`,
  move: `
    return fighters.map(f => {
      const s = submissions.find(x => x.actor === f.id);
      if (!s || f.hp <= 0) return { ...f };
      const step = s.plan.move === 'left' ? -1 : s.plan.move === 'right' ? 1 : 0;
      return { ...f, x: Math.max(0, Math.min(width - 1, f.x + step)) };
    });`,
  strike: `
    const hits = [];
    for (const s of submissions) {
      if (s.plan.action !== 'attack') continue;
      const a = moved.find(f => f.id === s.actor), t = moved.find(f => f.id === s.plan.target);
      if (!a || !t || a.hp <= 0 || a.cooldown > 0 || t.hp <= 0 || Math.abs(a.x - t.x) > 1) continue;
      const g = submissions.find(x => x.actor === t.id);
      hits.push({ attacker: a.id, target: t.id, amount: g && g.plan.action === 'guard' ? 1 : 2 });
    }
    return hits;`,
  wound: `return fighters.map(f => ({ ...f, hp: Math.max(0, f.hp - hits.filter(h => h.target === f.id).reduce((n, h) => n + h.amount, 0)) }));`,
  recover: `return fighters.map(f => ({ ...f, cooldown: hits.some(h => h.attacker === f.id) ? 1 : Math.max(0, f.cooldown - 1) }));`,
  planJudge: `
    const ok = plan.action === 'none' || plan.action === 'promise' && !!plan.detail || plan.action === 'give' && (observation.inventory[plan.item] ?? 0) >= 1;
    return { actor: observation.actor, ok, reason: ok ? 'a legal plan' : 'a give names an item the NPC holds' };`,
  story: `return 'Narrated ' + kind + ' with ' + events.length + ' events.';`,
};

/**
 * A model that plays every stage by its script. `policy` supplies the actors' own choices:
 * choose(actor) for merchants, tactic(actor) for fighters, respond/remember for NPCs. `corrupt` makes the first
 * (or every) natural-language settlement or resolution wrong, so that the commit's check is exercised.
 */
function gameModel({ choose = actor => TRADES[actor], tactic, respond, remember = () => [], corrupt = false } = {}) {
  const seen = [];
  const model = scriptedModel(opening => {
    const stage = STAGES.find(([phrase]) => opening.includes(phrase))?.[1];
    seen.push(stage); 
    if (stage === 'choose') return `return ${JSON.stringify(Object.fromEntries(['alice', 'bob', 'cara', 'dan'].map(id => [id, choose(id)])))}[observation.actor];`;
    if (stage === 'tactic') return `return ${JSON.stringify(Object.fromEntries(['a', 'b', 'c'].map(id => [id, tactic(id)])))}[observation.actor];`;
    if (stage === 'respond') return `return ${JSON.stringify(respond)};`;
    if (stage === 'remember') return `return ${JSON.stringify(remember())};`;
    if (corrupt === 'entries' && stage === 'settle')
      return `${SCRIPTS.settle.replace('return { outcomes, entries };', '')} entries.push({ kind: 'cash', from: ordered[0].actor, to: ordered[0].actor, amount: 1 }); return { outcomes, entries };`;
    if (corrupt && stage === 'settle')
      // A trade below the seller's price, with entries that match it: the settlement is consistent, the commit refuses it.
      return SCRIPTS.settle.replace('const made', 'if (outcome.status === "traded" && (problem === "" || ' + (corrupt === 'always') + ')) outcome.total -= 1;\n      const made');
    if (corrupt && stage === 'resolve')
      return `${SCRIPTS.resolve.replace('return { fighters: await recover(wounded, hits), hits, damage };', '')}
        const done = await recover(wounded, hits); if (problem === "" || ${corrupt === 'always'}) done[0].hp += 1; return { fighters: done, hits, damage };`;
    return SCRIPTS[stage] ?? null;
  });
  return { ...model, seen };
}

function play(scene, settings, options) {
  const model = gameModel(options);
  model.judged = withJudge(model.driver);
  const runtime = createNatlangRuntime({ model: model.driver, calls: false });
  return runtime.run(() => playTurn(scene, settings)).then(report => ({ report, model }));
}

const MERCHANTS = [
  { id: 'alice', cash: 10, goods: { apple: 0, bread: 0 }, offers: {} },
  { id: 'bob', cash: 0, goods: { apple: 2, bread: 0 }, offers: { apple: 3 } },
  { id: 'cara', cash: 0, goods: { apple: 0, bread: 2 }, offers: { bread: 2 } },
];
const balance = (state, id) => state.merchants.find(row => row.id === id);

// ---------------------------------------------------------------- economy

test('the seeded order is the SHA-256 order of seed, tick and actor', async () => {
  for (const text of ['', 'a', '33:0:alice', 'x'.repeat(55), 'x'.repeat(56), 'x'.repeat(64), 'é'.repeat(70)])
    assert.equal(await economyOrder.sha256(text), createHash('sha256').update(text).digest('hex'));
  const actors = ['alice', 'bob', 'cara', 'dan', 'eve'];
  const expected = [...actors].sort((a, b) => createHash('sha256').update(`33:2:${a}`).digest('hex').localeCompare(createHash('sha256').update(`33:2:${b}`).digest('hex')));
  assert.deepEqual(await economyOrder.default(33, 2, actors), expected);
  assert.notDeepEqual(await economyOrder.default(33, 3, actors), expected);
});

test('an economy tick in natural language settles in seeded order and conserves money and goods', async () => {
  const state = createEconomy(MERCHANTS, { seed: 33 });
  const { report, model } = await play({ kind: 'economy', state }, defaultSettings);
  assert.equal(report.ok, true, report.problem);
  assert.equal(report.state.tick, 1);
  assert.equal(balance(report.state, 'alice').cash, 4);
  assert.equal(balance(report.state, 'alice').goods.apple, 2);
  assert.equal(balance(report.state, 'bob').cash, 6);
  assert.equal(balance(report.state, 'bob').goods.apple, 0);
  assert.equal(report.events.at(-1).operation, 'economy.settle');
  assert.equal(report.events.at(-1).outcomes.find(row => row.actor === 'alice').status, 'traded');
  assert.equal(report.narration, 'Narrated economy with 4 events.');
  assert.equal(state.tick, 0, 'the snapshot is not changed by the turn');
  // Each merchant chose from its own observation; every NL part of the tick ran as a stage.
  assert.equal(model.seen.filter(stage => stage === 'choose').length, 3);
  assert.equal(model.seen.filter(stage => stage === 'judge').length, 3);
  assert.equal(model.seen.filter(stage => stage === 'quote').length, 3);
  assert.match(model.openings[model.seen.indexOf('choose')], /observation: MarketObservation|observation: \{/);
});

test('the crisp implementations settle the same tick the same way, whatever order the merchants are listed in', async () => {
  const natural = (await play({ kind: 'economy', state: createEconomy(MERCHANTS, { seed: 33 }) }, defaultSettings)).report;
  const crispRun = (await play({ kind: 'economy', state: createEconomy(MERCHANTS, { seed: 33 }) }, crispSettings)).report;
  const reversed = (await play({ kind: 'economy', state: createEconomy([...MERCHANTS].reverse(), { seed: 33 }) }, crispSettings)).report;
  assert.deepEqual(crispRun.state, natural.state);
  assert.deepEqual(crispRun.events, natural.events);
  assert.deepEqual(reversed.events.at(-1), crispRun.events.at(-1));
  assert.deepEqual([...reversed.state.merchants].sort((a, b) => a.id.localeCompare(b.id)), crispRun.state.merchants);
  assert.match(crispRun.narration, /alice bought 2 apple from bob for 6\./);
});

test('shadow mode runs both implementations of a pluggable part, serves the natural-language one and records agreement', async () => {
  const traces = [];
  const model = gameModel();
  withJudge(model.driver);
  const runtime = createNatlangRuntime({ model: model.driver, trace: trace => traces.push(trace) });
  const shadow = { validate: 'shadow', settle: 'shadow', resolve: 'shadow', remember: 'nl', narrate: 'nl' };
  const report = await runtime.run(() => playTurn({ kind: 'economy', state: createEconomy(MERCHANTS, { seed: 33 }) }, shadow));
  assert.equal(report.ok, true, report.problem);
  assert.equal(balance(report.state, 'alice').goods.apple, 2);
  const shadows = traces.flatMap(trace => trace.events).filter(event => event.kind === 'pluggable_shadow');
  assert.deepEqual(new Set(shadows.map(event => event.name)), new Set(['economy.validate', 'economy.settle']));
  assert.ok(shadows.length >= 4, 'one validation per merchant and one settlement');
  assert.ok(shadows.every(event => event.served === 'nl' && event.agree === true), JSON.stringify(shadows));
});

test('a contested good goes to the merchant the seed puts first, and the loser is rejected with a reason', async () => {
  const state = createEconomy([...MERCHANTS.slice(0, 2), { id: 'dan', cash: 10, goods: { apple: 0 }, offers: {} }, MERCHANTS[2]], { seed: 5 });
  const choose = actor => actor === 'alice' || actor === 'dan' ? { kind: 'buy', seller: 'bob', good: 'apple', quantity: 2 } : { kind: 'pass' };
  const { report } = await play({ kind: 'economy', state }, crispSettings, { choose });
  const outcomes = report.events.at(-1).outcomes;
  assert.deepEqual(outcomes.map(row => row.actor), await economyOrder.default(5, 0, ['alice', 'bob', 'cara', 'dan']));
  assert.equal(outcomes.filter(row => row.status === 'traded').length, 1);
  assert.equal(outcomes.find(row => row.status === 'rejected').reason, 'seller lacks the stock');
});

test('an invalid intent becomes a pass with the reason in the log; settlement never sees it', async () => {
  const choose = actor => actor === 'alice' ? { kind: 'buy', seller: 'alice', good: 'apple', quantity: 1 } : { kind: 'pass' };
  for (const settings of [crispSettings, defaultSettings]) {
    const { report } = await play({ kind: 'economy', state: createEconomy(MERCHANTS, { seed: 1 }) }, settings, { choose });
    assert.equal(report.ok, true);
    assert.match(report.log[0], /^alice: a buy names another merchant/);
    assert.equal(report.events.at(-1).outcomes.find(row => row.actor === 'alice').status, 'pass');
  }
});

test('a settlement the commit refuses is settled again with the problem; a second refusal leaves the tick undone', async () => {
  const state = createEconomy(MERCHANTS, { seed: 33 });
  const once = await play({ kind: 'economy', state }, defaultSettings, { corrupt: 'once' });
  assert.equal(once.report.ok, true);
  assert.equal(once.model.seen.filter(stage => stage === 'settle').length, 2);
  assert.equal(balance(once.report.state, 'bob').cash, 6);
  const always = await play({ kind: 'economy', state }, defaultSettings, { corrupt: 'always' });
  assert.equal(always.report.ok, false);
  assert.match(always.report.problem, /offered price/);
  assert.deepEqual(always.report.state, state, 'a failed tick leaves the world as it was');
  assert.match(always.report.narration, /^The turn did not happen: /);
});

test('the economy commit checks conservation, order, prices and the tick', async () => {
  const state = createEconomy(MERCHANTS, { seed: 33 });
  const submissions = (actors, intents = {}) => actors.map(actor => ({ actor, intent: intents[actor] ?? { kind: 'pass' } }));
  const sequence = await economyOrder.default(33, 0, ['alice', 'bob', 'cara']);
  const trade = { alice: { kind: 'buy', seller: 'bob', good: 'apple', quantity: 2 } };
  const traded = { actor: 'alice', status: 'traded', seller: 'bob', good: 'apple', quantity: 2, total: 6 };
  const settlementOf = outcome => ({ outcomes: sequence.map(actor => actor === 'alice' ? outcome : { actor, status: 'pass' }),
    entries: outcome.status === 'traded' ? [{ kind: 'good', from: 'bob', to: 'alice', good: 'apple', amount: outcome.quantity },
      { kind: 'cash', from: 'alice', to: 'bob', amount: outcome.total }] : [] });
  const effects = (outcome, over = {}) => ({ basis: 0, order: sequence, submissions: submissions(sequence, trade), settlement: settlementOf(outcome), ...over });
  assert.equal((await economyCommit(state, effects(traded))).ok, true);
  assert.match((await economyCommit(state, effects(traded, { basis: 1 }))).problem, /tick 1/);
  assert.match((await economyCommit(state, effects(traded, { order: [...sequence].reverse() }))).problem, /seeded order/);
  assert.match((await economyCommit(state, effects({ ...traded, total: 5 }))).problem, /offered price/);
  const three = { alice: { kind: 'buy', seller: 'bob', good: 'apple', quantity: 3 } };
  assert.match((await economyCommit(state, effects({ ...traded, quantity: 3, total: 9 }, { submissions: submissions(sequence, three) }))).problem, /below zero/);
  const minted = settlementOf(traded);
  minted.entries[1].amount = 7;
  assert.match((await economyCommit(state, effects(traded, { settlement: minted }))).problem, /entries are the goods-then-cash pair/);
  const broke = createEconomy([{ ...MERCHANTS[0], cash: 1 }, ...MERCHANTS.slice(1)], { seed: 33 });
  assert.match((await economyCommit(broke, effects(traded, { order: await economyOrder.default(33, 0, ['alice', 'bob', 'cara']) }))).problem, /below zero/);
  assert.match((await economyCommit(state, effects(traded, { submissions: submissions(sequence, { alice: { kind: 'buy', seller: 'nobody', good: 'apple', quantity: 1 } }) }))).problem, /seller/);
  // Money out of thin air keeps the entries consistent but not the totals.
  const forged = { ...state, initial: { ...state.initial, cash: state.initial.cash + 1 } };
  assert.match((await economyCommit(forged, effects(traded))).problem, /initial totals/);
});

test('the crisp settlement reference rejects with the reasons the natural-language stages give', () => {
  const ordered = [
    { actor: 'alice', intent: { kind: 'buy', seller: 'bob', good: 'pear', quantity: 1 } },
    { actor: 'cara', intent: { kind: 'buy', seller: 'bob', good: 'apple', quantity: 3 } },
    { actor: 'bob', intent: { kind: 'buy', seller: 'cara', good: 'bread', quantity: 1 } },
  ];
  const { outcomes } = economyReference(MERCHANTS, ordered);
  assert.deepEqual(outcomes.map(row => row.reason), ['seller does not offer the good', 'seller lacks the stock', 'buyer lacks the cash']);
});

// ---------------------------------------------------------------- combat

const ARENA = () => createArena([{ id: 'a', x: 1, hp: 5 }, { id: 'b', x: 3, hp: 5 }]);

test('natural-language combat resolves simultaneous movement and attacks, with cooldown as a timer', async () => {
  const tactic = actor => actor === 'a' ? { move: 'right', action: 'attack', target: 'b' } : { move: 'left', action: 'attack', target: 'a' };
  const { report, model } = await play({ kind: 'combat', state: ARENA() }, defaultSettings, { tactic });
  assert.equal(report.ok, true, report.problem);
  assert.deepEqual(report.state.fighters.map(row => [row.id, row.x, row.hp, row.cooldown]), [['a', 2, 3, 1], ['b', 2, 3, 1]]);
  assert.deepEqual(report.events[0].damage, { a: 2, b: 2 });
  for (const stage of ['move', 'strike', 'wound', 'recover']) assert.equal(model.seen.filter(seen => seen === stage).length, 1);
  // Cooldown blocks the next round's attack and then counts down; the following round's attack lands.
  const stay = actor => ({ move: 'stay', action: 'attack', target: actor === 'a' ? 'b' : 'a' });
  const second = (await play({ kind: 'combat', state: report.state }, defaultSettings, { tactic: stay })).report;
  assert.deepEqual(second.state.fighters.map(row => [row.hp, row.cooldown]), [[3, 0], [3, 0]]);
  const third = (await play({ kind: 'combat', state: second.state }, defaultSettings, { tactic: stay })).report;
  assert.deepEqual(third.state.fighters.map(row => [row.hp, row.cooldown]), [[1, 1], [1, 1]]);
  assert.equal(third.state.round, 3);
});

test('the crisp resolution gives the natural-language result: guard, clamping, the dead, and the line of reach', async () => {
  const state = createArena([{ id: 'a', x: 0, hp: 5 }, { id: 'b', x: 2, hp: 1 }, { id: 'c', x: 7, hp: 4 }], { width: 8 });
  const submissions = [
    { actor: 'a', plan: { move: 'left', action: 'attack', target: 'b' } },       // clamped at 0
    { actor: 'b', plan: { move: 'right', action: 'guard' } },                   // moves to 3: out of reach
    { actor: 'c', plan: { move: 'right', action: 'rest' } },                    // clamped at 7
  ];
  const reference = combatReference(state.fighters, state.width, submissions);
  assert.deepEqual(reference.hits, []);
  assert.deepEqual(reference.fighters.map(row => row.x), [0, 3, 7]);
  const hitting = [{ actor: 'a', plan: { move: 'right', action: 'attack', target: 'b' } },
    { actor: 'b', plan: { move: 'left', action: 'guard' } }, submissions[2]];
  const guarded = combatReference(state.fighters, state.width, hitting);
  assert.deepEqual(guarded.hits, [{ attacker: 'a', target: 'b', amount: 1 }]);
  assert.deepEqual(guarded.fighters.map(row => [row.x, row.hp, row.cooldown]), [[1, 5, 1], [1, 0, 0], [7, 4, 0]]);
  // The same round played through the natural-language stages.
  const tactics = { a: hitting[0].plan, b: hitting[1].plan, c: hitting[2].plan };
  const model = gameModel({ tactic: id => tactics[id] });
  withJudge(model.driver);
  const runtime = createNatlangRuntime({ model: model.driver, calls: false });
  const nl = await runtime.run(() => playTurn({ kind: 'combat', state }, { ...defaultSettings, validate: 'crisp' }));
  assert.deepEqual(nl.state.fighters, guarded.fighters);
  assert.equal(nl.state.round, 1);
});

test('an attack on oneself is invalid and becomes stay and rest', async () => {
  const tactic = actor => actor === 'a' ? { move: 'stay', action: 'attack', target: 'a' } : { move: 'stay', action: 'rest' };
  const { report } = await play({ kind: 'combat', state: ARENA() }, crispSettings, { tactic });
  assert.equal(report.ok, true);
  assert.match(report.log[0], /^a: an attack names another living fighter/);
  assert.deepEqual(report.events[0].damage, {});
});

test('a fighter who is down submits nothing', async () => {
  const state = { ...ARENA(), fighters: [{ id: 'a', x: 1, hp: 5, cooldown: 0 }, { id: 'b', x: 2, hp: 0, cooldown: 0 }] };
  const { model, report } = await play({ kind: 'combat', state }, crispSettings, { tactic: () => ({ move: 'stay', action: 'rest' }) });
  assert.equal(model.seen.filter(stage => stage === 'tactic').length, 1);
  assert.equal(report.state.round, 1);
});

test('a resolution the commit refuses is resolved again with the problem; a second refusal leaves the round undone', async () => {
  const tactic = () => ({ move: 'stay', action: 'rest' });
  const once = await play({ kind: 'combat', state: ARENA() }, defaultSettings, { tactic, corrupt: 'once' });
  assert.equal(once.report.ok, true);
  assert.equal(once.model.seen.filter(stage => stage === 'resolve').length, 2);
  const always = await play({ kind: 'combat', state: ARENA() }, defaultSettings, { tactic, corrupt: 'always' });
  assert.equal(always.report.ok, false);
  assert.match(always.report.problem, /health falls by exactly the damage taken/);
  assert.equal(always.report.state.round, 0);
});

test('the combat commit checks reach, damage, movement and timers', () => {
  const state = ARENA();
  const plans = [{ actor: 'a', plan: { move: 'stay', action: 'attack', target: 'b' } }, { actor: 'b', plan: { move: 'stay', action: 'rest' } }];
  const good = combatReference(state.fighters, state.width, plans);
  assert.equal(good.hits.length, 0, 'two cells apart is out of reach');
  assert.equal(combatCommit(state, { basis: 0, submissions: plans, resolution: good }).ok, true);
  const mutate = change => { const copy = structuredClone(good); change(copy); return combatCommit(state, { basis: 0, submissions: plans, resolution: copy }); };
  assert.match(mutate(r => { r.hits.push({ attacker: 'a', target: 'b', amount: 2 }); r.damage = { b: 2 }; r.fighters[1].hp = 3; }).problem, /within one cell/);
  assert.match(mutate(r => { r.fighters[0].x = 3; }).problem, /at most one cell/);
  assert.match(mutate(r => { r.fighters[1].hp = 4; }).problem, /exactly the damage taken/);
  assert.match(mutate(r => { r.fighters[0].cooldown = 3; }).problem, /cooldown/);
  assert.match(mutate(r => { r.fighters[0].x = -1; }).problem, /inside the arena/);
  assert.match(mutate(r => { r.fighters.reverse(); }).problem, /same fighters in the same order/);
  assert.match(combatCommit(state, { basis: 4, submissions: plans, resolution: good }).problem, /round 4/);
  assert.match(combatCommit(state, { basis: 0, submissions: [plans[0], plans[0]], resolution: good }).problem, /at most one tactic/);
});

// ---------------------------------------------------------------- NPCs

const VILLAGE = () => createVillage([{ id: 'innkeeper', inventory: { key: 1 } }, { id: 'guard', inventory: { key: 0 } }]);
const ASK = { kind: 'npc', actor: 'innkeeper', event: { from: 'guest', text: 'May I have a key?' } };

test('an NPC acts from its own memory: the event and its notes are remembered, the action is exact', async () => {
  const respond = { say: 'Here is a key.', action: 'give', item: 'key', target: 'guest' };
  const remember = () => [{ text: 'The guest wants a room key.', about: ['event-1'] }];
  const { report, model } = await play({ ...ASK, state: VILLAGE() }, defaultSettings, { respond, remember });
  assert.equal(report.ok, true, report.problem);
  const innkeeper = report.state.actors.find(row => row.id === 'innkeeper');
  assert.equal(innkeeper.inventory.key, 0);
  assert.deepEqual(innkeeper.memory.map(row => row.id), ['event-1', 'event-1.note-1']);
  assert.deepEqual(innkeeper.memory[1].about, ['event-1']);
  assert.deepEqual(report.state.applied, ['event-1']);
  assert.equal(report.state.actors.find(row => row.id === 'guard').memory.length, 0);
  assert.equal(report.events[0].action, 'give');
  assert.match(model.openings[model.seen.indexOf('respond')], /observation: NpcObservation|observation: \{/);
  // A later event sees the earlier memory, and numbers its own entry next.
  const follow = await play({ ...ASK, state: report.state, event: { from: 'guest', text: 'Thank you.' } }, defaultSettings, { respond: { say: 'Welcome.', action: 'none' } });
  assert.deepEqual(follow.report.state.actors[0].memory.map(row => row.id), ['event-1', 'event-1.note-1', 'event-2']);
  assert.match(follow.model.openings[follow.model.seen.indexOf('respond')], /May I have a key/);
});

test('a promise records its evidence; crisp remembering keeps the event alone', async () => {
  const respond = { say: 'I will help.', action: 'promise', target: 'guest', detail: 'Find the map' };
  const { report, model } = await play({ ...ASK, state: VILLAGE() }, crispSettings, { respond });
  assert.deepEqual(report.state.actors[0].commitments, [{ to: 'guest', detail: 'Find the map', evidence_id: 'event-1' }]);
  assert.deepEqual(report.state.actors[0].memory.map(row => row.id), ['event-1']);
  assert.equal(model.seen.includes('remember'), false);
});

test('a plan the NPC cannot carry out keeps the speech and takes no action', async () => {
  const respond = { say: 'Here is a key.', action: 'give', item: 'key', target: 'guest' };
  const state = VILLAGE();
  const { report } = await play({ kind: 'npc', state, actor: 'guard', event: ASK.event }, crispSettings, { respond });
  assert.equal(report.ok, true);
  assert.match(report.log[0], /^guard: a give names an item the NPC holds/);
  assert.equal(report.events[0].said, 'Here is a key.');
  assert.equal(report.events[0].action, 'none');
});

test('an event is acted on once, and forged effects are refused', () => {
  const state = VILLAGE();
  const effects = { basis: 0, actor: 'innkeeper', event_id: 'event-1', event: ASK.event, notes: [],
    plan: { say: 'Hello', action: 'give', item: 'key', target: 'guest' } };
  const first = npcCommit(state, effects);
  assert.equal(first.ok, true);
  assert.equal(first.state.version, 1);
  assert.match(npcCommit(first.state, effects).problem, /version 0|already/);
  assert.match(npcCommit(first.state, { ...effects, basis: 1 }).problem, /already/);
  assert.match(npcCommit(first.state, { ...effects, basis: 1, event_id: 'event-2' }).problem, /holds/);
  assert.match(npcCommit(state, { ...effects, event_id: 'event-9' }).problem, /next event is event-1/);
  assert.match(npcCommit(state, { ...effects, actor: 'guard' }).problem, /guard: a give names an item the NPC holds/);
  assert.match(npcCommit(state, { ...effects, notes: [{ text: 'x', about: ['event-77'] }] }).problem, /cites entries of its memory/);
  assert.match(npcCommit(state, { ...effects, actor: 'nobody' }).problem, /invalid NPC event/);
  assert.deepEqual(state, VILLAGE(), 'the commit does not change the state it was given');
});

// ---------------------------------------------------------------- host

test('a session moves its state only when a turn commits, and refuses a turn decided on an old state', async () => {
  const session = new Session(createEconomy(MERCHANTS, { seed: 33 }), crispSettings, async (scene, settings) => (await play(scene, settings)).report);
  const first = session.turn();
  const second = session.turn();
  await first;
  await assert.rejects(second, /stale turn/);
  assert.equal(session.state.tick, 1);
  assert.equal(session.events.at(-1).operation, 'economy.settle');
  const village = new Session(VILLAGE(), crispSettings, async (scene, settings) => (await play(scene, settings, { respond: { say: 'Hi', action: 'none' } })).report);
  await assert.rejects(village.turn(), /needs an actor and an event/);
  assert.equal((await village.turn({ actor: 'guard', event: ASK.event })).ok, true);
});

test('initial worlds are validated at the boundary', () => {
  assert.throws(() => createEconomy([MERCHANTS[0]]), /invalid economy/);
  assert.throws(() => createEconomy([MERCHANTS[0], MERCHANTS[0]]), /duplicate/);
  assert.throws(() => createEconomy([MERCHANTS[0], { ...MERCHANTS[1], cash: -1 }]), /invalid merchant/);
  assert.throws(() => createArena([{ id: 'a', x: 9, hp: 1 }, { id: 'b', x: 1, hp: 1 }]), /invalid fighters/);
  assert.throws(() => createVillage([{ id: 'x y', inventory: {} }]), /invalid NPCs/);
  assert.deepEqual(createEconomy(MERCHANTS).initial, { cash: 10, goods: { apple: 2, bread: 2 } });
});

// ---------------------------------------------------------------- refinements

/**
 * The model of gameModel, except that a rejection by a refinement is answered with `fix`, the code of a corrected
 * value; the feedback the model was sent is kept in `feedback`.
 */
function repairing(options, fix) {
  const base = gameModel(options), feedback = [];
  const driver = Object.assign(async request => {
    const last = request.messages.at(-1);
    if (last.role === 'tool' && /refinement-unsatisfied/.test(String(last.content))) {
      feedback.push(String(last.content));
      return { calls: [['eval', { code: fix }]] };
    }
    return base.driver(request);
  }, {});
  base.judged = withJudge(driver, options?.truth);
  return { ...base, driver, feedback };
}

function playWith(model, scene, settings, extra = {}) {
  const traces = [];
  const runtime = createNatlangRuntime({ model: model.driver, calls: false, refinements: { ...extra }, trace: trace => traces.push(trace) });
  return runtime.run(() => playTurn(scene, settings)).then(report => ({ report, traces }));
}
const checks = traces => traces.flatMap(trace => trace.events).filter(event => event.kind === 'refinement_check');

test('every predicate of types.ts has a crisp checker', () => {
  const types = readFileSync(new URL('../../applications/games/types.ts', import.meta.url), 'utf8');
  const declared = new Set([...types.matchAll(/Is<[^"]*"([^"]*)"/g)].map(match => match[1].replace(/\s+/g, ' ').trim()));
  const keys = new Set(Object.keys(CRISP));
  assert.deepEqual([...declared].filter(key => !keys.has(key)), []);
  assert.deepEqual([...keys].filter(key => !declared.has(key)), []);
});

test('crisp checkers decide the exact predicates; a whole tick in natural language needs no judge', async () => {
  assert.equal(CRISP["a name: a letter followed by letters, digits, '_' or '-'"]('x y'), false);
  assert.equal(CRISP['a positive safe integer'](0), false);
  assert.equal(CRISP['a non-negative safe integer'](1.5), false);
  assert.equal(CRISP['0 or 1'](2), false);
  assert.equal(CRISP['1 or 2'](2), true);
  const { report, model } = await play({ kind: 'economy', state: createEconomy(MERCHANTS, { seed: 33 }) }, defaultSettings);
  assert.equal(report.ok, true, report.problem);
  assert.deepEqual(model.judged, [], 'every refined slot of the tick has a crisp checker');
});

test('a quantity of zero is sent back with the refinement error and repaired, without the judge', async () => {
  const bad = { kind: 'buy', seller: 'bob', good: 'apple', quantity: 0 };
  const model = repairing({ choose: actor => actor === 'alice' ? bad : { kind: 'pass' } }, `return ${JSON.stringify(TRADES.alice)};`);
  const { report, traces } = await playWith(model, { kind: 'economy', state: createEconomy(MERCHANTS, { seed: 33 }) }, crispSettings);
  assert.equal(report.ok, true, report.problem);
  assert.equal(balance(report.state, 'alice').goods.apple, 2);
  assert.equal(model.feedback.length, 1);
  assert.match(model.feedback[0], /a positive safe integer/);
  assert.deepEqual(model.judged, []);
  assert.ok(checks(traces).some(event => event.outcome === 'fail' && event.source === 'crisp'));
});

test('a settlement whose entries are not the traded outcomes is repaired in the stage, before the commit sees it', async () => {
  const model = repairing({ corrupt: 'entries' }, SCRIPTS.settle);
  const { report, traces } = await playWith(model, { kind: 'economy', state: createEconomy(MERCHANTS, { seed: 33 }) }, defaultSettings);
  assert.equal(report.ok, true, report.problem);
  assert.equal(balance(report.state, 'bob').cash, 6);
  assert.match(model.feedback[0], /goods from seller to buyer and then the money from buyer to seller/);
  assert.equal(model.seen.filter(stage => stage === 'settle').length, 1, 'one settle call: the commit never had to ask again');
  assert.ok(checks(traces).every(event => event.source === 'crisp' || event.source === 'cache'));
});

test('settings that name shadow for a part with one implementation are refused before any stage runs', async () => {
  const model = gameModel();
  withJudge(model.driver);
  const runtime = createNatlangRuntime({ model: model.driver, calls: false });
  await assert.rejects(runtime.run(() => playTurn({ kind: 'economy', state: createEconomy(MERCHANTS) }, { ...defaultSettings, narrate: 'shadow' })),
    error => error.code === 'refinement-unsatisfied' && /remember and narrate are each nl or crisp/.test(error.message));
  assert.deepEqual(model.seen, []);
});
