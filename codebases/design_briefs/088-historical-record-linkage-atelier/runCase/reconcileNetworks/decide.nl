---
description: Decide field-level findings for reconcileNetworks.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Check proposed identity and relationship links globally for impossible dates, incompatible simultaneous identities, and circular assumptions. Prevent one uncertain link from becoming evidence that confirms itself through a derived family or institutional network. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
