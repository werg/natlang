/*---
description: Invoke the host's exact natlang type grammar and fit checks.
args:
  target: Target
  context: Context
  candidate: Candidate
returns: FitReport
effects: [types.check]
---*/
return fx.types.check(args.target, args.context, args.candidate);
