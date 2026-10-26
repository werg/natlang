---
args: { candidate: ResearchAttempt, decisions: DecisionsNegativeFinding, evidence: EvidencePage[] }
returns: DraftNegativeFinding
---
1. Construct an evidence-backed NegativeFinding record from established atoms, using the fields declared for NegativeFinding in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Determine whether evidence supports a negative result, an invalid measurement, an unfinished procedure, or genuine uncertainty. Preserve the distinction even when all cases were informally described as failures and avoid stronger conclusions than the attempt warrants.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
