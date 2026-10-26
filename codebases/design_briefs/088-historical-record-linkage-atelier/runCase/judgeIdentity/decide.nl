---
description: Decide field-level findings for judgeIdentity.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare evidence for identity versus kinship, institutional succession, or coincidence. Weigh source reliability only within its actual scope, preserve competing candidates, and avoid treating a plausible story as proof of identity. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
