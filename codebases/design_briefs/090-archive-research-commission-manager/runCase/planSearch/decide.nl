---
description: Decide field-level findings for planSearch.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Select collections and search strategies using catalog structure, terminology variants, dates, and known access limits. Include negative search paths and record why a collection is relevant rather than relying only on keyword matches. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
