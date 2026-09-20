/*---
description: Atomically validate and apply a semantic spell plan to a pure arena snapshot.
args:
  observation: World
  rules: Rules
  meaning: Interpretation
returns: CastResult
---*/
const original = args.observation;
const meaning = args.meaning;
const fail = (status, explanation) => ({ status, world: original, cost: 0, explanation });
if (meaning.status !== "plan") return fail(meaning.status, meaning.explanation);
const plan = meaning.plan;
if (plan.revision !== original.revision) return fail("stale", "The arena changed before this plan could apply.");
if (original.revision >= Number.MAX_SAFE_INTEGER) return fail("invalid", "The arena revision cannot advance safely.");
if (!plan.effects.length) return fail("invalid", "A cast needs at least one effect.");
if (plan.effects.length > 8) return fail("invalid", "A cast may contain at most eight effects.");
const r = args.rules;
const limits = [r.max_damage, r.max_shield, r.max_move, r.damage_cost, r.shield_cost, r.move_cost];
if (limits.some(n => !Number.isSafeInteger(n) || n < 0) ||
    !Number.isSafeInteger(original.energy) || original.energy < 0 ||
    !Number.isSafeInteger(original.revision) || original.revision < 0)
  return fail("invalid", "The arena rules or resources are invalid.");
const actors = original.actors.map(a => ({ ...a }));
if (new Set(actors.map(a => a.id)).size !== actors.length ||
    actors.some(a => !Number.isSafeInteger(a.hp) || a.hp < 0 ||
                     !Number.isSafeInteger(a.shield) || a.shield < 0 ||
                     !Number.isSafeInteger(a.x) || !Number.isSafeInteger(a.y)))
  return fail("invalid", "The arena actor state is invalid.");
let cost = 0;
for (const effect of plan.effects) {
  const target = actors.find(a => a.id === effect.target);
  if (!target) return fail("invalid", `Unknown target ${effect.target}.`);
  if (!Number.isSafeInteger(effect.amount) || !Number.isSafeInteger(effect.x) || !Number.isSafeInteger(effect.y))
    return fail("invalid", "Effects need integer parameters.");
  if (effect.kind === "damage") {
    if (effect.amount < 1 || effect.amount > r.max_damage || target.hp === 0)
      return fail("invalid", "Damage is outside the rules or the target is already down.");
    cost += effect.amount * r.damage_cost;
    const absorbed = Math.min(target.shield, effect.amount);
    target.shield -= absorbed;
    target.hp = Math.max(0, target.hp - (effect.amount - absorbed));
  } else if (effect.kind === "shield") {
    if (effect.amount < 1 || effect.amount > r.max_shield || target.hp === 0)
      return fail("invalid", "Shielding is outside the rules or the target is already down.");
    cost += effect.amount * r.shield_cost;
    target.shield += effect.amount;
  } else if (effect.kind === "move") {
    const distance = Math.abs(effect.x - target.x) + Math.abs(effect.y - target.y);
    if (effect.amount !== 0 || distance < 1 || distance > r.max_move || target.hp === 0 ||
        actors.some(a => a.id !== target.id && a.hp > 0 && a.x === effect.x && a.y === effect.y))
      return fail("invalid", "Movement is outside the rules or the destination is occupied.");
    cost += distance * r.move_cost;
    target.x = effect.x;
    target.y = effect.y;
  } else return fail("invalid", "Unknown effect kind.");
  if (!Number.isSafeInteger(cost)) return fail("invalid", "The cast cost overflowed.");
}
if (cost > original.energy) return fail("insufficient", "The caster lacks energy for the whole spell.");
return { status: "applied", world: { revision: original.revision + 1, energy: original.energy - cost, actors },
         cost, explanation: meaning.explanation };
