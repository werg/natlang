---
description: Decide field-level findings for recoverLearningMechanism.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Analyze the original training to identify the skill exercised by each scenario and the elements that merely provide context. Separate essential decision structure from names, settings, idioms, or assumptions that may be adapted without changing the learning objective. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
