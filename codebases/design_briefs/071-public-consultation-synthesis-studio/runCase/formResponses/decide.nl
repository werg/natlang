---
description: Decide field-level findings for formResponses.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Draft evidence-linked response options for the accountable public body, including clarification, further investigation, or policy alternatives. Distinguish what the consultation establishes from questions requiring external evidence and avoid choosing political priorities not supplied by decision-makers. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
