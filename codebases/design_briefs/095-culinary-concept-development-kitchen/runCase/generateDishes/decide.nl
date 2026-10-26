---
description: Decide field-level findings for generateDishes.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Invent distinct dishes using available ingredients and approved culinary procedures, explaining flavor, texture, and service intent. Propose substitutions only when their properties and dietary implications are verified, and preserve specialist food-safety review where needed. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
