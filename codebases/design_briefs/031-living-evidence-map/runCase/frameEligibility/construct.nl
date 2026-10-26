---
args: { candidate: EvidenceReview, decisions: DecisionsEligibilityCriterion, evidence: EvidencePage[] }
returns: DraftEligibilityCriterion
---
1. Construct an evidence-backed EligibilityCriterion record from established atoms, using the fields declared for EligibilityCriterion in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Translate the research question into explicit screening criteria and distinguish mandatory eligibility from useful descriptive metadata. Define ambiguous boundaries with examples before full screening and preserve questions requiring expert methodological judgment.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
