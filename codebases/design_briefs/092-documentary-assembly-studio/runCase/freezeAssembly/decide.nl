---
description: Decide field-level findings for freezeAssembly.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Produce a versioned edit decision list, script, asset manifest, and unresolved-work register. Verify exact timecodes and duration, bind approvals to the assembly, and invalidate affected checks when a source clip or sequence changes. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
