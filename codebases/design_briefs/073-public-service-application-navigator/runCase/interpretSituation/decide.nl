---
description: Decide field-level findings for interpretSituation.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read the person’s stated situation and supplied documents to identify needs and established facts. Avoid inferring sensitive characteristics, distinguish missing information from ineligibility, and route urgent concerns according to approved service procedures. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
