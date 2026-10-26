---
description: Decide field-level findings for simulatePlayers.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Run exact state-space checks and simulated player interpretations, including out-of-order discovery, lost props, repeated actions, and plausible wrong answers. Identify soft locks and confusing clues, then propose targeted repairs rather than adding arbitrary hints everywhere. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
