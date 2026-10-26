---
description: Decide field-level findings for constructArguments.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Develop alternative narrative structures around the editorial question and available evidence. Include relevant contradictions and uncertainty, distinguish thematic juxtaposition from causal implication, and avoid inventing connective facts to make the story more dramatic. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
