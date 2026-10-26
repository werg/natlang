---
description: Decide field-level findings for reviewProduction.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Review the proposed assembly for factual support, rights, attribution, accessibility, and unresolved editorial issues. Distinguish verified claims from attributed testimony and commission targeted fact-checking or clearance tasks before locking the edit. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
