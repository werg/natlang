---
description: Decide field-level findings for verifyEdition.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Re-run continuity checks over affected scenes and downstream knowledge states, inspect arc payoffs, and assemble a coherent editorial diff. Record approved canon changes and preserve prior editions so future work uses the correct story-world state. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
