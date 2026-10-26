---
description: Decide field-level findings for modelNarrativeInformation.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read the script and approved scene metadata to identify dialogue, sound, visual information, and narrative reveals. Distinguish essential understanding from decorative detail and preserve what the audience is meant to know at each moment. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
