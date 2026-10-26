---
description: Decide field-level findings for prepareResearchView.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Publish a reviewable linkage view with source-level explanations and explicit alternatives. Support corrections and reversible splits, preserve stable source identifiers, and expose uncertainty to downstream researchers instead of exporting only a deceptively clean graph. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
