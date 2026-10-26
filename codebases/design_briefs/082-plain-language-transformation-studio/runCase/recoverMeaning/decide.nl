---
description: Decide field-level findings for recoverMeaning.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Parse the source into claims, duties, permissions, conditions, and exceptions before rewriting. Identify terms that require explanation and preserve uncertainty or scope; do not treat dense wording as permission to remove consequential distinctions. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
