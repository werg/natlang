---
args: { candidate: ExperimentDesign, decisions: DecisionsMeasurementFinding, evidence: EvidencePage[] }
returns: DraftMeasurementFinding
---
1. Construct an evidence-backed MeasurementFinding record from established atoms, using the fields declared for MeasurementFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare metric definitions, event schemas, instrumentation notes, and missingness reports. Identify whether treatment changed the measurement process itself; formulate exact diagnostic queries instead of assuming a numerical difference reflects changed customer behavior.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
