---
description: Choose the next operator of the adaptation episode.
args:
  history: AdaptationState["history"]
  plan: RevisionPlan
  remainingSteps: number
returns: Operator
---
Choose the operator for the next revision. "learned-proposal" asks the learned optimizer for a complete bundle,
"local-refinement" improves the current bundle by gradient on the support evidence, "retrieval" mixes the bank's
components without gradients, and "keep" makes no change.

1. When every operation in plan.operations is "keep", return "keep".
2. When history is empty, return "learned-proposal".
3. When the last revision was accepted, return "learned-proposal".
4. When the last two revisions were learned proposals and both were rejected, return "local-refinement".
5. When the last revision was a rejected local refinement, return "retrieval".
6. Otherwise return "learned-proposal".
