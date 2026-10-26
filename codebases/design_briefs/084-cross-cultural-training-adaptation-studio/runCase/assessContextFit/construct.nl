---
description: Construct a ContextFitFinding draft for assessContextFit.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TrainingMechanism[] }
returns: DraftContextFitFinding
---
Assemble the ContextFitFinding as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Adaptation rationale is evidence-based and audience-specific, not stereotyped.
