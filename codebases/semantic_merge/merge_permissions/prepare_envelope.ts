export default function prepare_envelope(revision: number, updates: Update[]): Prepared {
const invalid = error => ({ valid: false, error, updates: updates, presentation: "" });
if (!Number.isSafeInteger(revision) || revision < 0) return invalid("Invalid base revision.");
if (updates.length > 16) return invalid("More than 16 finite deliveries.");
const byId = new Map();
for (const update of updates) {
  if (!update.id.trim() || update.base_revision !== revision ||
      new Set(update.parents).size !== update.parents.length)
    return invalid(`Invalid update ID, parents or base revision: ${update.id}`);
  const normalized = { ...update, parents: [...update.parents].sort() };
  const previous = byId.get(update.id);
  if (previous) {
    const same = previous.base_revision === normalized.base_revision &&
                 previous.author === normalized.author && previous.text === normalized.text &&
                 JSON.stringify(previous.parents) === JSON.stringify(normalized.parents);
    if (!same) return invalid(`Conflicting deliveries reuse update ID ${update.id}.`);
  }
  byId.set(update.id, normalized);
}
for (const update of byId.values())
  if (update.parents.some(parent => parent === update.id || !byId.has(parent)))
    return invalid(`Missing or self-referential parent for ${update.id}.`);
const pending = new Map(byId);
const ordered = [];
for (const _round of [...pending.keys()]) {
  const ready = [...pending.values()].filter(u => u.parents.every(p => !pending.has(p)))
    .sort((a, b) => a.id < b.id ? -1 : a.id > b.id ? 1 : 0)[0];
  if (!ready) return invalid("Causal parent cycle.");
  ordered.push(ready);
  pending.delete(ready.id);
}
return { valid: true, error: "", updates: ordered,
         presentation: JSON.stringify(ordered.map(u => u.id)) };
}
