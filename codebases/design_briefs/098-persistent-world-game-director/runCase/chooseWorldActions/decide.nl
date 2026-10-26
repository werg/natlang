---
description: Decide field-level findings for chooseWorldActions.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Choose among exact legal actions in a way consistent with the character’s knowledge, goals, and relationships. Generate dialogue and creative alternatives within world and content boundaries, and avoid overriding resource or location constraints for narrative convenience. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
