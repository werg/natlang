---
args: { candidate: EligibilityCriterion, decisions: DecisionsStudyIdentity, evidence: EvidencePage[] }
returns: DraftStudyIdentity
---
1. Construct an evidence-backed StudyIdentity record from established atoms, using the fields declared for StudyIdentity in the application types.
2. Preserve identifiers and source versions. Do not promote a disputed or unknown value to a fact; leave absent values absent and retain alternatives in conflicts.
3. Carry the domain distinction through construction: Compare authors, enrollment windows, populations, identifiers, and methods to determine whether reports describe the same study or overlapping cohorts. Preserve uncertain links and distinguish follow-up analyses from independent replications before counting evidence.
4. Return the draft record with its exact source references and conflicts. Host code owns identity joins, calculations, authorization, persistence, and effects.
