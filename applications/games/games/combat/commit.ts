/**
 * The arena's commit: a pure, bounded application of a round's effects to its state, and the check of what the
 * stages produced. Movement is at most one cell inside the arena, health falls by exactly the damage taken, every
 * hit is legal on the state it was judged on, timers never rise without a strike. Returns the next state and the
 * events, or the state unchanged and the problem.
 */
import rules from './validate/rules.js';
import type { CombatEffects, CombatState, Committed } from '../../types.js';

const same = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b);
const whole = (value: unknown) => Number.isSafeInteger(value) && (value as number) >= 0;

export default function commit(state: CombatState, effects: CombatEffects): Committed {
  const refuse = (problem: string): Committed => ({ ok: false, state, events: [], problem });
  const { submissions, resolution } = effects;
  if (effects.basis !== state.round) return refuse(`the effects are for round ${effects.basis}; the arena is at round ${state.round}`);
  const actors = submissions.map(row => row.actor);
  if (new Set(actors).size !== actors.length) return refuse('each fighter submits at most one tactic');
  for (const submission of submissions) {
    const verdict = rules(state, submission);
    if (!verdict.ok) return refuse(`${submission.actor}: ${verdict.reason}`);
  }
  const plans = new Map(submissions.map(row => [row.actor, row.plan]));
  const after = resolution.fighters;
  if (!same(after.map(row => row.id), state.fighters.map(row => row.id))) return refuse('the round returns the same fighters in the same order');
  const dealt: Record<string, number> = {};
  for (const hit of resolution.hits) dealt[hit.target] = (dealt[hit.target] ?? 0) + hit.amount;
  if (!same(Object.entries(dealt).sort(), Object.entries(resolution.damage).sort())) return refuse('damage is the sum of the hits on each target');
  const struck = new Set<string>();
  for (let i = 0; i < state.fighters.length; i++) {
    const before = state.fighters[i]!;
    const now = after[i]!;
    if (!whole(now.hp) || !whole(now.cooldown) || !Number.isSafeInteger(now.x)) return refuse(`${before.id}: x, hp and cooldown are whole numbers, hp and cooldown at least 0`);
    if (now.x < 0 || now.x >= state.width) return refuse(`${before.id}: x stays inside the arena`);
    if (Math.abs(now.x - before.x) > 1) return refuse(`${before.id}: a fighter moves at most one cell`);
    if ((before.hp <= 0 || !plans.has(before.id)) && now.x !== before.x) return refuse(`${before.id}: only a living fighter with a tactic moves`);
    if (now.hp !== Math.max(0, before.hp - (dealt[before.id] ?? 0))) return refuse(`${before.id}: health falls by exactly the damage taken, to a floor of 0`);
  }
  const beforeOf = new Map(state.fighters.map(row => [row.id, row]));
  const afterOf = new Map(after.map(row => [row.id, row]));
  for (const hit of resolution.hits) {
    const attacker = beforeOf.get(hit.attacker), target = beforeOf.get(hit.target), plan = plans.get(hit.attacker);
    if (!attacker || !target || plan?.action !== 'attack' || plan.target !== hit.target) return refuse(`${hit.attacker}: a hit follows an attack on its target`);
    if (attacker.hp <= 0 || attacker.cooldown > 0) return refuse(`${hit.attacker}: only a living fighter off cooldown lands a hit`);
    if (target.hp <= 0) return refuse(`${hit.attacker}: a hit lands on a living target`);
    if (Math.abs(afterOf.get(hit.attacker)!.x - afterOf.get(hit.target)!.x) > 1) return refuse(`${hit.attacker}: a hit lands within one cell after movement`);
    if (hit.amount !== (plans.get(hit.target)?.action === 'guard' ? 1 : 2)) return refuse(`${hit.attacker}: a hit does 2 damage, or 1 against a guard`);
    if (struck.has(hit.attacker)) return refuse(`${hit.attacker}: a fighter lands at most one hit per round`);
    struck.add(hit.attacker);
  }
  for (const before of state.fighters) {
    const now = afterOf.get(before.id)!;
    if (struck.has(before.id) ? now.cooldown < 1 : now.cooldown > before.cooldown) return refuse(`${before.id}: a cooldown starts at a strike and otherwise only counts down`);
  }
  const next: CombatState = { ...state, round: state.round + 1, fighters: structuredClone(after) };
  const events = [{ operation: 'combat.resolve', round: next.round, fighters: structuredClone(after), damage: { ...resolution.damage } }];
  return { ok: true, state: next, events, problem: '' };
}
