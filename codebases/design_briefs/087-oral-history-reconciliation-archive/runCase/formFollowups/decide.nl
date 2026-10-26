---
description: Decide field-level findings for formFollowups.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Generate neutral follow-up questions that could clarify identities, chronology, or context without leading speakers toward a preferred account. Respect consent and participation boundaries, and distinguish clarification from challenging a person’s memory as false. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
