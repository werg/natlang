export default function prepare_step(base: Document, _current: MergeResult, _update: Update): Step {
const current = _current;
const update = _update;
const ids = current.updates.map(u => u.id);
const claims = [...current.applied, ...current.alternatives.flatMap(a => a.update_ids)];
const invalid = error => ({ kind: "invalid", error, updates: [...current.updates, update],
                            presentation: JSON.stringify([...ids, update.id]) });
if (current.status === "rejected" || current.base_revision !== base.revision ||
    ids.length !== new Set(ids).size || claims.length !== ids.length ||
    new Set(claims).size !== ids.length || claims.some(id => !ids.includes(id)))
  return invalid("The prior merge state has invalid or incomplete provenance.");
if (!update.id.trim() || update.base_revision !== base.revision ||
    new Set(update.parents).size !== update.parents.length)
  return invalid("The new update has an invalid ID, parent list or base revision.");
const earlier = current.updates.find(u => u.id === update.id);
if (earlier) {
  const same = earlier.base_revision === update.base_revision && earlier.author === update.author &&
               earlier.text === update.text &&
               JSON.stringify([...earlier.parents].sort()) === JSON.stringify([...update.parents].sort());
  return same ? { kind: "duplicate", error: "", updates: current.updates,
                  presentation: current.presentation } : invalid(`Conflicting delivery reuses ${update.id}.`);
}
if (update.parents.some(parent => !ids.includes(parent)))
  return invalid(`The new update has a missing causal parent: ${update.id}.`);
if (current.updates.length >= 16) return invalid("Incremental history exceeds 16 updates.");
return { kind: "new", error: "", updates: [...current.updates, update],
         presentation: JSON.stringify([...ids, update.id]) };
}
