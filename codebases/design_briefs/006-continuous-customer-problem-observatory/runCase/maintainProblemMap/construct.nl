---
args: { candidate: IncidentGroup, decisions: DecisionsProblemConcept, evidence: EvidencePage[] }
returns: DraftProblemConcept
---
1. Construct an evidence-backed ProblemConcept record from established atoms, using the fields declared for ProblemConcept in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Cluster by the failed outcome and mechanism rather than shared vocabulary. Propose new concepts when existing boundaries do not fit, preserve meaningful subtypes, and review counterexamples before moving historical episodes between concepts.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
