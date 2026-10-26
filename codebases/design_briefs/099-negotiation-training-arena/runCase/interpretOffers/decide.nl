---
description: Decide field-level findings for interpretOffers.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Parse natural-language offers into issue-level terms, conditions, and commitment status. Distinguish exploration from a firm offer and qualified acceptance from final agreement; ask clarifying questions when ambiguity would change the negotiated outcome. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
