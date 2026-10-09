import type { Committed, EconomyEffects, EconomyState } from "../../types.js";
/**
 * The economy's commit: a pure, bounded application of a tick's effects to its state, and the check of what the
 * stages produced. Conservation (money and each good keep their initial totals, no balance goes below zero) and
 * determinism (the order is the seeded order, the entries are exactly the traded outcomes' entries, a trade is at
 * the seller's offered price). Returns the next state and the events, or the state unchanged and the problem.
 */
import { post, entriesOf, lowest, totalsOf } from './ledger.js';
import order from './order.js';
import rules from './validate/rules.js';

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);

export default function commit(state: EconomyState, effects: EconomyEffects): Committed {
  const refuse = (problem: string): Committed => ({ ok: false, state, events: [], problem });
  const { submissions, settlement } = effects;
  const ids = state.merchants.map(row => row.id);
  if (effects.basis !== state.tick) return refuse(`the effects are for tick ${effects.basis}; the economy is at tick ${state.tick}`);
  const actors = submissions.map(row => row.actor);
  if (new Set(actors).size !== actors.length) return refuse('each merchant submits at most one intent');
  for (const submission of submissions) {
    const verdict = rules(ids, submission);
    if (!verdict.ok) return refuse(`${submission.actor}: ${verdict.reason}`);
  }
  const expected = order(state.seed, state.tick, actors);
  if (!same(effects.order, expected) || !same(submissions.map(row => row.actor), expected))
    return refuse('the intents settle in the seeded order of this seed and tick');
  if (!same(settlement.outcomes.map(row => row.actor), expected)) return refuse('there is one outcome per intent, in the seeded order');
  if (!same(settlement.entries, settlement.outcomes.flatMap(entriesOf)))
    return refuse('the entries are the goods-then-cash pair of each traded outcome, in order');
  let balances = state.merchants;
  for (const [i, outcome] of settlement.outcomes.entries()) {
    const intent = submissions[i]!.intent;
    if (intent.kind === 'pass' ? outcome.status !== 'pass' : outcome.status === 'pass')
      return refuse(`${outcome.actor}: a pass settles as pass and a buy as traded or rejected`);
    if (outcome.status === 'traded') {
      const price = state.merchants.find(row => row.id === outcome.seller)?.offers[outcome.good!];
      if (outcome.seller !== intent.seller || outcome.good !== intent.good || outcome.quantity !== intent.quantity)
        return refuse(`${outcome.actor}: a trade is the seller, good and quantity of the intent`);
      if (!Number.isSafeInteger(price) || outcome.total !== outcome.quantity! * price!)
        return refuse(`${outcome.actor}: a trade is at the seller's offered price`);
    }
    for (const entry of entriesOf(outcome)) {
      balances = post(balances, [entry]);
      const below = lowest(balances);
      if (below) return refuse(`${outcome.actor}: no balance goes below zero, but ${below}`);
    }
  }
  if (!same(totalsOf(balances), state.initial)) return refuse('money and each good keep their initial totals');
  const next: EconomyState = { ...state, tick: state.tick + 1, merchants: balances };
  const events = [
    ...submissions.map(row => ({ operation: 'economy.intent', tick: state.tick, actor: row.actor, intent: structuredClone(row.intent) })),
    { operation: 'economy.settle', tick: state.tick, outcomes: structuredClone(settlement.outcomes) },
  ];
  return { ok: true, state: next, events, problem: '' };
}
