---
description: Decide field-level findings for interpretCircumstances.
args: { goal: string; slice: EvidenceSlice }
returns: DecisionAtom[]
---
Apply this domain algorithm to the prepared slice: Read the patient’s stated practical circumstances and communication preferences without inferring diagnoses or capability. Identify logistical barriers and questions requiring staff attention, and verify authority before involving a representative. For each consequential field return a value and evidence IDs, mark established, disputed, unknown, or inapplicable, and name a plausible alternative. Do not turn missing evidence into a negative fact or treat repeated volume as independent support unless the domain rule says so.
