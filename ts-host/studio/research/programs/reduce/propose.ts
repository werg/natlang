/*---
engine: typescript-host
args:
  base: Text
  edits: Edit[]
  removes: Text[]
  effect_ids: Text[]
  message: Text
returns: CandidateResult
---*/
return await host.research.propose(args.base, args.edits, args.removes, args.effect_ids, args.message);
