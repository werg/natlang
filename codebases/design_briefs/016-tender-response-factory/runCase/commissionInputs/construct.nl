---
args: { candidate: CapabilityMatch, decisions: DecisionsContributionTask, evidence: EvidencePage[] }
returns: DraftContributionTask
---
1. Construct an evidence-backed ContributionTask record from established atoms, using the fields declared for ContributionTask in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Group missing technical, financial, and implementation contributions by responsible team. Ask precise questions tied to requirements, specify acceptable evidence, and expose dependencies that could make the submission infeasible before the deadline.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
