---
description: Decide field-level findings for interpretIntent.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Interpret player language in the visible world context and map it to permitted action types. Distinguish speech, intention, and executed action, ask only consequential clarifications, and avoid granting capabilities because a player narrates success confidently. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
