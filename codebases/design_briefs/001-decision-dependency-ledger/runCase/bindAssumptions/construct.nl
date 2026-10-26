---
args: { candidate: DecisionAssertion, decisions: DecisionsAssumptionBinding, evidence: EvidencePage[] }
returns: DraftAssumptionBinding
---
1. Construct an evidence-backed AssumptionBinding record from established atoms, using the fields declared for AssumptionBinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: For each decision ask what would have to change to make reconsideration necessary. Separate essential premises from background rationale; emit a directed dependency with a counterfactual explanation and a passage establishing the connection.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
