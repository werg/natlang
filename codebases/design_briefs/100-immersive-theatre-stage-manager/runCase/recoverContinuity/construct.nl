---
description: Construct a TheatreRecovery draft for recoverContinuity.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: TheatreCuePlan[] }
returns: DraftTheatreRecovery
---
Assemble the TheatreRecovery as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Recovery preserves safety, consent, timing feasibility, and narrative continuity.
