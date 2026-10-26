---
args: { candidate: WorkStatus, decisions: DecisionsHandoverDependency, evidence: EvidencePage[] }
returns: DraftHandoverDependency
---
1. Construct an evidence-backed HandoverDependency record from established atoms, using the fields declared for HandoverDependency in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Identify knowledge, access, approvals, and external relationships needed to finish each item. Separate credentials from instructions, detect circular handoffs, and formulate minimal access requests without copying secrets into the handover dossier.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
