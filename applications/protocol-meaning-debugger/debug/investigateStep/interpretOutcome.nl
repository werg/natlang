---
description: Interpret a probe result under captured mechanisms and controls.
args:
  context: InvestigationContext
  hypotheses: Hypothesis[]
  candidate: ProbeCandidate
returns: ProbeOutcome
---
1. Capture targeted mechanisms, changed premises, unchanged source roles, baseline and candidate
   intermediate plans and exact receipts. Keep target/left/right identities distinct throughout comparison.
2. In eval declare local OutcomeJudgment with classification: 'discriminating' | 'coincidental' |
   'inadmissible' | 'unresolved', supportedIds:string[], rejectedIds:string[], inconclusiveIds:string[],
   reason:string, witnesses:{sourceId:string;path:string;quote:string}[]. Construct nl.with<OutcomeJudgment>
   over changing input this candidate's actual materialized trial receipts and role-specific plan differences.
3. Its instructions are: judge which captured mechanism this intervention exercises; require the
   predicted causal row/anchor/numerator change rather than headline disagreement alone; preserve
   confounded mechanisms as inconclusive and explain a genuine numerical coincidence separately.
4. Directly await the literal. Consume returned ID sets to update hypothesis status; validate every
   named ID belongs to the supplied hypothesis inventory.
5. Return seedId, prediction, classification, support/rejection/inconclusive sets, exact witnesses and
   localDecision with this site and actual returned reason. The reducer consumes this payload before shrinking.
6. A polarity-only seed with unchanged membership can support metric divergence. A shifted calibration
   seed needs changed window membership evidence to support anchor divergence.
7. If the observed computation contradicts the predicted transformation, classify unresolved and retain
   both receipts. Exactness exposes a model/service interpretation bug rather than proving a new causal story.
8. Do not score the structured result as one decision token. The actual typed output drives later
   investigation choices and the final root-cause witness.
