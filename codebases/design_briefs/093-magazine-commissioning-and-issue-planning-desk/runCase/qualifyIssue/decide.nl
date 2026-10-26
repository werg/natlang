---
description: Decide field-level findings for qualifyIssue.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Reconcile final pieces, captions, headlines, credits, layout, and publication timing as one issue. Check that promotional language does not overstate article claims, preserve contingency plans for late withdrawals, and freeze the approved edition. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
