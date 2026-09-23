---
description: Propose a TS-style natlang signature from bounded source and caller
  evidence. Read files for source context when the bounded snapshot leaves a
  question open.
args:
  target: Target
  context: Context
returns: Assessment
---
function infer(target, context) -> Assessment
  candidate = propose(target, context)
  fit = check_candidate(target, context, candidate)
  return finalize(candidate, fit)
