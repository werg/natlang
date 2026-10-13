---
description: Build source-backed mechanisms that counterfactual trials can distinguish.
args:
  contrasts: Contrast[]
  baseline: PlanState
returns: Hypothesis[]
---
1. For every different or unresolved Contrast, state the specific mechanism that could change rows,
   event offsets or the counted proposition. Give it the visible hypothesis ID if the contract declares one.
2. A population mechanism identifies which eligibility predicate differs. An anchor mechanism identifies
   the event choice that moves inclusion windows. A metric mechanism identifies numerator polarity.
3. Attach affectedDefinitionIds and the contrast witnesses. Hypothesis identity is not a arbitrary
   synonym for a probe name; multiple probes can supply evidence about the same mechanism.
4. Mark source-explicit differences possible until their predicted consequences are independently
   exercised. Baseline agreement does not reject them, and baseline disagreement alone does not prove them.
5. Keep unresolved source bindings as unresolved hypotheses with the precise missing clause. They
   need evidence about ownership, not another numerical calculation using a guessed definition.
6. Preserve interactions: a sample shift and polarity reversal can compensate on one table. A probe
   that changes both does not isolate either mechanism without its intermediate membership receipts.
7. Return the finite hypothesis inventory. Source-equivalent roles remain controls, not manufactured
   additional failure mechanisms intended to make the application appear more complex.
8. Later updates reference these exact IDs and returned probe contributions. Never introduce a new
   explanation after observing a result without attaching it to source evidence and an explicit contrast.
