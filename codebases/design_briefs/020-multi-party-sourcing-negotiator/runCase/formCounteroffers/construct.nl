---
args: { candidate: SourcingPackage, decisions: DecisionsCounteroffer, evidence: EvidencePage[] }
returns: DraftCounteroffer
---
1. Construct an evidence-backed Counteroffer record from established atoms, using the fields declared for Counteroffer in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Draft conditional counteroffers that exchange permitted concessions for improvements in the whole package. Keep supplier-confidential information isolated and make dependencies explicit so acceptance of one term cannot accidentally commit the buyer to an unapproved package.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
