---
args: { prior: SubstitutionCheckpoint|null, request: SubstituteEvaluationRequest, catalog: SupplierCatalogSnapshot, candidates: StageResultSubstituteCandidate|null, compatibility: StageResultCompatibilityFinding|null, qualification: StageResultQualificationTask|null, packet: StageResultSubstitutionPacket|null }
returns: SubstitutionCheckpoint
---
Bind the checkpoint to the catalog edition and intended machine/environment. Preserve each candidate supplier revision. A revision change invalidates only compatibility findings using the old revision and packets containing that candidate. Keep lot acceptance, adapter readiness and use approval as separate evidence; none substitutes for another. Store qualification receipt identities once and reopen only failed or unverified cases. An identical evidence edition is a no-op; a changed supplier revision or receipt advances the checkpoint.
