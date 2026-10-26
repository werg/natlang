---
description: Decide field-level findings for validateLocale.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Run exact placeholder, markup, length, plural-form, and link checks, then review semantic fidelity and naturalness independently. Inspect cross-segment consistency and context screenshots or metadata, distinguishing harmless variation from contradictory terminology or behavior. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
