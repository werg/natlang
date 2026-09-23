export default function finish_step(current: MergeResult, update: Update, step: Step, draft: Draft): MergeResult {
const ids = step.updates.map(u => u.id);
const claims = [...draft.applied, ...draft.alternatives.flatMap(a => a.update_ids)];
const complete = claims.length === ids.length && new Set(claims).size === ids.length &&
                 claims.every(id => ids.includes(id)) &&
                 draft.alternatives.every(a => a.update_ids.length > 0 && a.reason.trim());
if (!complete) return {
  status: "rejected", text: current.text, applied: current.applied,
  alternatives: [...current.alternatives,
    { update_ids: [update.id], proposal: update.text,
      reason: "The incremental draft did not account for every update exactly once." }],
  explanation: "Incomplete incremental provenance.", base_revision: current.base_revision,
  updates: step.updates, presentation: step.presentation
};
return { status: draft.alternatives.length ? "unresolved" : "merged", text: draft.text,
         applied: draft.applied, alternatives: draft.alternatives,
         explanation: draft.explanation, base_revision: current.base_revision,
         updates: step.updates, presentation: step.presentation };
}
