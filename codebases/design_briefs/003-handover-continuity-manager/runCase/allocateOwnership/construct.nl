---
args: { candidate: HandoverDependency, decisions: DecisionsOwnershipTransfer, evidence: EvidencePage[] }
returns: DraftOwnershipTransfer
---
1. Construct an evidence-backed OwnershipTransfer record from established atoms, using the fields declared for OwnershipTransfer in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare successor remit, stated capacity, and required capabilities. Propose explicit ownership with acceptance criteria and unresolved obligations; a recipient mentioned in a meeting is not treated as accepting the transfer.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
