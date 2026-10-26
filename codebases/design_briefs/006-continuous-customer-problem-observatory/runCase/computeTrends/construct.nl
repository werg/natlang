---
args: { candidate: ProblemConcept, decisions: DecisionsTrendSlice, evidence: EvidencePage[] }
returns: DraftTrendSlice
---
1. Construct an evidence-backed TrendSlice record from established atoms, using the fields declared for TrendSlice in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Define an observable population and comparable time windows before invoking exact aggregation. Interpret changes in collection channels and exposure; distinguish a rise in complaints from a rise in the complaint rate among eligible customers.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
