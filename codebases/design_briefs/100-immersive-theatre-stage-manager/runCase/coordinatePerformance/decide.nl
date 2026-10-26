---
description: Decide field-level findings for coordinatePerformance.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Build an exact cue and movement plan using approved production constraints, then provide role-specific briefings. Confirm critical acknowledgments, preserve stage-manager control, and distinguish a proposed cue from an executed cue in the live state. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
