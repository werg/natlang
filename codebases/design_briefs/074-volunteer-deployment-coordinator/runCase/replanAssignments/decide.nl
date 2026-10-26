---
description: Decide field-level findings for replanAssignments.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Process withdrawals and changed mission conditions by identifying exact capability gaps. Propose replacement teams or reduced scope, seek fresh consent for changed roles, and cancel unsafe or incomplete deployments rather than pretending nominal headcount is sufficient. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
