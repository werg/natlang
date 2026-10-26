---
description: Decide field-level findings for locateRecords.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Search authorized inventories using verified patient and episode identifiers, preserving date and category boundaries. Distinguish unavailable records from records outside scope and prevent similar identities or merged historical accounts from contaminating the packet. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
