---
description: Decide field-level findings for identifyPathways.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Compare known facts with supplied service descriptions and conditions to identify potentially relevant routes. Preserve multiple plausible pathways, explain uncertainty, and avoid presenting a preliminary match as an official eligibility decision. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
