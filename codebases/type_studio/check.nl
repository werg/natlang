import assemble from "./check/assemble";
import check_calls from "./check/check_calls";
import identify_calls from "./check/identify_calls";
---
description: Explore possible calls in a natlang body and separate witnessed
  mismatches from hypotheses. Read files for source context when the bounded
  snapshot leaves a question open.
args:
  target: Target
  context: Context
returns: CheckReport
effects:
  - types.calls
---
function check(target, context) -> CheckReport
  claims = identify_calls(target, context)
  diagnostics = check_calls(context, claims)
  return assemble(claims, diagnostics)
