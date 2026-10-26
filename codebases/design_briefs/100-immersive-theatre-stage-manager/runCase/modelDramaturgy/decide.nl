---
description: Decide field-level findings for modelDramaturgy.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read the script and production notes to identify scene goals, knowledge reveals, dependencies, and branching choices. Distinguish fixed dramatic obligations from improvisable material and preserve what each character and audience group can know at that point. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
