---
description: Decide field-level findings for assessCompatibility.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare each proposed act with supplied rights and consent conditions, including interactions among multiple rightsholders. Preserve scoped exceptions and refer ambiguous legal interpretation to the rights officer rather than selecting the most permissive reading. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
