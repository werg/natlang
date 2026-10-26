---
args: { candidate: ContractCase, decisions: DecisionsContractObligation, evidence: EvidencePage[] }
returns: DraftContractObligation
---
1. Construct an evidence-backed ContractObligation record from established atoms, using the fields declared for ContractObligation in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Extract atomic obligations with actor, beneficiary, trigger, deadline, and exceptions. Separate descriptions and aspirations from operative provisions, retain cross-references, and preserve unknown roles rather than assigning responsibility from nearby names.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
