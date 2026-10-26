---
description: Decide field-level findings for rebuildRunOfShow.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Use exact scheduling and resource checks to construct a revised run of show with owners, dependencies, and decision deadlines. Reserve resources before announcing changes, preserve crew readiness, and identify points beyond which a contingency is no longer feasible. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
