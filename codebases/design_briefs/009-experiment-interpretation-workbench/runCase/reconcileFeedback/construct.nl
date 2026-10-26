---
args: { candidate: ResultClaim, decisions: DecisionsMechanismFinding, evidence: EvidencePage[] }
returns: DraftMechanismFinding
---
1. Construct an evidence-backed MechanismFinding record from established atoms, using the fields declared for MechanismFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare qualitative accounts with result patterns to formulate plausible mechanisms, not causal proof. Seek dissatisfied successes and satisfied failures that challenge the simple explanation, and identify evidence capable of distinguishing rival accounts.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
