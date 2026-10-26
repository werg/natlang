---
description: Decide field-level findings for constructTermOptions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Propose terms and scoped usage rules considering approved domain conventions, grammatical forms, and audience needs. Explain tradeoffs among clarity, continuity, and precision, and retain regional or contextual variants where one universal term would change meaning. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
