---
description: Decide field-level findings for mapParticipants.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Identify transport providers, equipment suppliers, caregivers, and receiving services with their actual responsibilities. Distinguish availability from acceptance, verify representation and information-sharing authority, and avoid assuming family relationships imply consent or capacity. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
