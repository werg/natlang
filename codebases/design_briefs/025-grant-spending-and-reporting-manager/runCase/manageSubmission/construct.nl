---
args: { candidate: GrantNarrative, decisions: DecisionsGrantSubmission, evidence: EvidencePage[] }
returns: DraftGrantSubmission
---
1. Construct an evidence-backed GrantSubmission record from established atoms, using the fields declared for GrantSubmission in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Assemble the required forms, schedules, and evidence, obtain accountable approval, and record actual submission receipt. Track follow-up questions as obligations and revise reports through explicit versions rather than silently replacing the submitted record.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
