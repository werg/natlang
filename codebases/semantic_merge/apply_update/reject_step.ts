/*---
args:
  current: MergeResult
  update: Update
  step: Step
returns: MergeResult
---*/
return { status: "rejected", text: args.current.text, applied: args.current.applied,
         alternatives: [...args.current.alternatives,
           { update_ids: [args.update.id], proposal: args.update.text, reason: args.step.error }],
         explanation: args.step.error, base_revision: args.current.base_revision,
         updates: args.step.updates, presentation: args.step.presentation };
