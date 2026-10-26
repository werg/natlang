---
description: Decide field-level findings for maintainUseRegister.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Record approved uses, required attribution, expiry, and downstream distribution. Recheck conditions before release, schedule removals or renewals where needed, and propagate consent changes to affected derivatives and public surfaces. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
