/*---
engine: typescript-host
args:
  head: Text
  edits: Edit[]
  removes: Text[]
  effect_ids: Text[]
  message: Text
returns: CommitResult
---*/
return await host.research.commit(args.head, args.edits, args.removes, args.effect_ids, args.message);
