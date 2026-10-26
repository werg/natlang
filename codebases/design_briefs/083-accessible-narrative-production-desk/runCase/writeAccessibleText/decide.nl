---
description: Decide field-level findings for writeAccessibleText.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Formulate concise, natural accessible text appropriate to the channel and supplied style guidance. Describe observable actions rather than inferred motives, identify speakers or meaningful sounds where needed, and preserve narrative tone without adding unsupported interpretation. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
