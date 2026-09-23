export default function finish(base: Document, prepared: Prepared, draft: Draft): MergeResult {
const ids = prepared.updates.map(u => u.id);
const claims = [...draft.applied, ...draft.alternatives.flatMap(a => a.update_ids)];
const complete = claims.length === ids.length && new Set(claims).size === ids.length &&
                 claims.every(id => ids.includes(id)) &&
                 draft.alternatives.every(a => a.update_ids.length > 0 && a.reason.trim());
if (!complete) return {
  status: "rejected", text: base.text, applied: [],
  alternatives: prepared.updates.map(u => ({ update_ids: [u.id], proposal: u.text,
                                                 reason: "The semantic draft did not account for every update exactly once." })),
  explanation: "The semantic draft had incomplete provenance.", base_revision: base.revision,
  updates: prepared.updates, presentation: prepared.presentation
};
return { status: draft.alternatives.length ? "unresolved" : "merged", text: draft.text,
         applied: draft.applied, alternatives: draft.alternatives,
         explanation: draft.explanation, base_revision: base.revision,
         updates: prepared.updates, presentation: prepared.presentation };
}
