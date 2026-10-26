---
args: { artifacts: ReleaseArtifact[], selections: SpanSelection[], plan: RedactionPlan, approvals: RecipientApprovalReview, challenges: EvidencePage[] }
returns: DisclosureChallengeMap
---
For each challenge in the supplied pages, identify the exact document, artifact hash and challenged locator. Match the artifact to selections by documentId and originalHash; trace the challenged allowed or withheld span to its actual sourceRefs and the plan's redaction reason. Record a released approval binding only if approvals contains that exact renderedHash and an approvalRef. Unknown challenge targets, missing source locators, disputed redaction reasons and absent approval remain unresolved.

Build mappings only from matched challenge/artifact/source/approval evidence. Preserve each mapping's challengeRef, documentId, artifactHash, sourceRefs, redactionReason and approvalRefs; an empty approvalRefs means no release authority was established. Bind withheldDocumentIds to documents with an unresolved challenge that could disclose restricted content or falsify the retained meaning. Bind unresolvedChallenges to the specific missing evidence or disputed reading. Return DisclosureChallengeMap { mappings, withheldDocumentIds, unresolvedChallenges }. No challenge record mutates a previously released artifact.
