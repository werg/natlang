export default function prepare(base: Document, updates: Update[]): Prepared {
const invalid = error => ({ valid: false, error, updates: updates, presentation: "" });
if (!Number.isSafeInteger(base.revision) || base.revision < 0)
  return invalid("Invalid base revision.");
if (updates.length > 16) return invalid("History exceeds the finite merge limit of 16 deliveries.");
const byId = new Map();
for (const update of updates) {
  if (!update.id || !update.id.trim() || !Number.isSafeInteger(update.base_revision) ||
      update.base_revision !== base.revision || new Set(update.parents).size !== update.parents.length)
    return invalid(`Invalid update identity, parents or base revision: ${update.id}`);
  const earlier = byId.get(update.id);
  if (earlier) {
    const same = earlier.base_revision === update.base_revision && earlier.author === update.author &&
                 earlier.text === update.text &&
                 JSON.stringify([...earlier.parents].sort()) === JSON.stringify([...update.parents].sort());
    if (!same) return invalid(`Conflicting deliveries reuse update ID ${update.id}.`);
  } else byId.set(update.id, { ...update, parents: [...update.parents].sort() });
}
for (const update of byId.values())
  if (update.parents.some(parent => parent === update.id || !byId.has(parent)))
    return invalid(`Missing or self-referential causal parent for ${update.id}.`);
const remaining = new Map(byId);
const ordered = [];
while (remaining.size) {
  const next = [...remaining.values()].filter(u => u.parents.every(p => !remaining.has(p)))
    .sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)[0];
  if (!next) return invalid("Causal parent cycle.");
  ordered.push(next);
  remaining.delete(next.id);
}
return { valid: true, error: "", updates: ordered,
         presentation: JSON.stringify(ordered.map(u => u.id)) };
}
