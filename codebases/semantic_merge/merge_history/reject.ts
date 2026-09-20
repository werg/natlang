/*---
args:
  base: Document
  prepared: Prepared
returns: MergeResult
---*/
return { status: "rejected", text: args.base.text, applied: [],
         alternatives: args.prepared.updates.map(u => ({ update_ids: [u.id], proposal: u.text,
                                                        reason: args.prepared.error })),
         explanation: args.prepared.error, base_revision: args.base.revision,
         updates: args.prepared.updates, presentation: args.prepared.presentation };
