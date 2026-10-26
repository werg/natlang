---
description: Decide field-level findings for interpretProposals.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Translate informal proposals into clear intended outcomes, work scope, and affected groups. Preserve the proposer’s intent, distinguish a preferred implementation from the underlying benefit, and request clarification where ambiguity changes feasibility or eligibility. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
