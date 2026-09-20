/*---
description: Enforce complete provenance after one incremental semantic reduction.
args:
  current: MergeResult
  update: Update
  step: Step
  draft: Draft
returns: MergeResult
---*/
const ids = args.step.updates.map(u => u.id);
const claims = [...args.draft.applied, ...args.draft.alternatives.flatMap(a => a.update_ids)];
const complete = claims.length === ids.length && new Set(claims).size === ids.length &&
                 claims.every(id => ids.includes(id)) &&
                 args.draft.alternatives.every(a => a.update_ids.length > 0 && a.reason.trim());
if (!complete) return {
  status: "rejected", text: args.current.text, applied: args.current.applied,
  alternatives: [...args.current.alternatives,
    { update_ids: [args.update.id], proposal: args.update.text,
      reason: "The incremental draft did not account for every update exactly once." }],
  explanation: "Incomplete incremental provenance.", base_revision: args.current.base_revision,
  updates: args.step.updates, presentation: args.step.presentation
};
return { status: args.draft.alternatives.length ? "unresolved" : "merged", text: args.draft.text,
         applied: args.draft.applied, alternatives: args.draft.alternatives,
         explanation: args.draft.explanation, base_revision: args.current.base_revision,
         updates: args.step.updates, presentation: args.step.presentation };
