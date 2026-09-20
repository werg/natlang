/*---
args:
  base: Document
  prepared: Prepared
returns: MergeResult
---*/
return { status: "merged", text: args.base.text, applied: [], alternatives: [],
         explanation: "No updates.", base_revision: args.base.revision,
         updates: [], presentation: args.prepared.presentation };
