import type { CombatResolution, CombatSubmission, Fighter, Hit } from '../../../types.js';

/** The crisp resolution of a round. The reference of resolve/policy.nl. */
export default function reference(fighters: Fighter[], width: number, submissions: CombatSubmission[]): CombatResolution {
  const plans = new Map(submissions.map(row => [row.actor, row.plan]));
  // 1. Movement: every living submitter steps at most one cell, inside the arena.
  const moved: Fighter[] = fighters.map(row => {
    const plan = plans.get(row.id);
    if (!plan || row.hp <= 0) return { ...row };
    const step = plan.move === 'left' ? -1 : plan.move === 'right' ? 1 : 0;
    return { ...row, x: Math.max(0, Math.min(width - 1, row.x + step)) };
  });
  // 2. Strikes: against positions after movement and health and cooldown from before the round.
  const hits: Hit[] = [];
  for (const attacker of moved) {
    const plan = plans.get(attacker.id);
    const target = plan?.target ? moved.find(row => row.id === plan.target) : undefined;
    if (!plan || plan.action !== 'attack' || attacker.hp <= 0 || attacker.cooldown > 0 || !target || target.hp <= 0 ||
        Math.abs(attacker.x - target.x) > 1) continue;
    hits.push({ attacker: attacker.id, target: target.id, amount: plans.get(target.id)?.action === 'guard' ? 1 : 2 });
  }
  const damage: Record<string, number> = {};
  for (const hit of hits) damage[hit.target] = (damage[hit.target] ?? 0) + hit.amount;
  // 3. Wounds and 4. cooldown timers.
  const struck = new Set(hits.map(hit => hit.attacker));
  const after = moved.map(row => ({ ...row, hp: Math.max(0, row.hp - (damage[row.id] ?? 0)),
    cooldown: struck.has(row.id) ? 1 : Math.max(0, row.cooldown - 1) }));
  return { fighters: after, hits, damage };
}
