---
args: { prior: DisclosureCheckpoint|null, request: DisclosureReviewRequest, evidence: DisclosureEvidencePack, scope: DisclosureScopeDecision, selections: SpanSelection[], sanitized: RedactionPlan, artifacts: ReleaseArtifact[], approvals: RecipientApprovalReview, challengeMap: DisclosureChallengeMap }
returns: DisclosureCheckpoint
---
Bind history to the authorized recipient, purpose and case. Preserve source hashes, exact sanitization and render receipts, and recipient approvals by artifact hash. A recipient or purpose change invalidates prior release approval; changed rendered bytes require fresh approval. Keep withheld document IDs and unresolved challenges until new evidence resolves their exact blockers. Record dispatch only from an actual receipt for the approved hash and recipient. Identical source and receipt replay is a no-op.
