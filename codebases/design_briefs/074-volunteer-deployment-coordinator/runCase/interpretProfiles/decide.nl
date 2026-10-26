---
description: Decide field-level findings for interpretProfiles.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read volunteer profiles and confirmations to distinguish self-reported skills, verified qualifications, availability, and accepted roles. Preserve limitations and preferences, and avoid treating past participation as consent to a new assignment. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
