/*---
description: Check complete provenance of a semantic draft, retaining every source update on rejection.
args:
  base: Document
  prepared: Prepared
  draft: Draft
returns: MergeResult
---*/
const ids = args.prepared.updates.map(u => u.id);
const claims = [...args.draft.applied, ...args.draft.alternatives.flatMap(a => a.update_ids)];
const complete = claims.length === ids.length && new Set(claims).size === ids.length &&
                 claims.every(id => ids.includes(id)) &&
                 args.draft.alternatives.every(a => a.update_ids.length > 0 && a.reason.trim());
if (!complete) return {
  status: "rejected", text: args.base.text, applied: [],
  alternatives: args.prepared.updates.map(u => ({ update_ids: [u.id], proposal: u.text,
                                                 reason: "The semantic draft did not account for every update exactly once." })),
  explanation: "The semantic draft had incomplete provenance.", base_revision: args.base.revision,
  updates: args.prepared.updates, presentation: args.prepared.presentation
};
return { status: args.draft.alternatives.length ? "unresolved" : "merged", text: args.draft.text,
         applied: args.draft.applied, alternatives: args.draft.alternatives,
         explanation: args.draft.explanation, base_revision: args.base.revision,
         updates: args.prepared.updates, presentation: args.prepared.presentation };
