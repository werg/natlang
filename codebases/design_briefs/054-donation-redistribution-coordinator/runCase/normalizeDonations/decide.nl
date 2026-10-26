---
args: { prepared: PreparedEvidence, goal: GoalContract, policy: PolicySelection }
returns: DecisionAtom[]
---
Apply this domain algorithm to prepared evidence: Interpret offers and condition evidence, distinguishing available items from tentative promises. Record quantities, collection constraints, and any permitted-use restrictions; request missing information that changes suitability rather than accepting vague offers into inventory.
For each field, return a value only when supported, source IDs, authority, one rejected alternative, and status established, disputed, unknown, or inapplicable. Preserve contrary evidence and distinguish observation from inference.
