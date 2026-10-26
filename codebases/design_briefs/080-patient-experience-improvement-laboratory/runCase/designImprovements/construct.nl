---
description: Construct a ServiceExperiment draft for designImprovements.
args: { goal: string; slice: EvidenceSlice; decisions: DecisionAtom[]; prior: ServiceMechanism[] }
returns: DraftServiceExperiment
---
Assemble the ServiceExperiment as a field-level draft from established decisions only. Preserve qualifiers and links to prior records. Carry disputed alternatives into conflicts and missing fields into unresolved. Do not invent IDs, dates, quantities, authority, or effects. The domain artifact is used to support this gate: Experiment does not alter clinical care and has an independent outcome measure.
