---
description: Build a purpose-limited disclosure from reviewed source spans, verified sanitization receipts, and a rendered output bound to the approved recipient.
args: { request: DisclosureReviewRequest, evidence: DisclosureEvidencePack }
returns: DisclosureReleaseProposal
uses: [disclosureDesk/resolveDisclosureScope, disclosureDesk/selectContestedSpans, disclosureDesk/mapCombinedInference, disclosureDesk/preserveQualifierContext, disclosureDesk/sanitizeTextAndMetadata, disclosureDesk/verifyRenderedArtifact, disclosureDesk/checkRecipientApproval, disclosureDesk/mapChallengeToSources]
---
1. Bind scope = resolveDisclosureScope(request, evidence.cases, evidence.policyCandidates). If authority, recipient, purpose, or date window is unresolved, withhold affected documents before span review.
2. Bind selections = selectContestedSpans(scope, evidence.documentInspections). Retain each source hash and locator; exclude unrelated spans with a recorded reason.
3. Bind plan = mapCombinedInference(selections, evidence.documentInspections). Evaluate re-identification across the assembled fragments, not only within individual pages.
4. Bind qualifiedPlan = preserveQualifierContext(selections, plan, evidence.documentInspections). Retain essential qualifiers; if a qualifier itself is restricted, withhold the statement and state the limitation.
5. Bind sanitized = sanitizeTextAndMetadata(qualifiedPlan, evidence.redactionReceipts, evidence.documentInspections). Require an exact hash-bound receipt covering visible content, OCR text layers, and embedded metadata; an unverified redaction is withheld.
6. Bind artifacts = verifyRenderedArtifact(sanitized, evidence.renderReceipts). Compare sanitized output hash to rendered output and re-extracted text; residual marks or hidden text require a new artifact and fresh render.
7. Bind approvals = checkRecipientApproval(request, evidence.recipientApprovals). Any recipient, purpose, document or artifact-hash change invalidates old delivery authorization.
8. Bind challengeMap = mapChallengeToSources(artifacts, evidence.challengeMappings). Preserve an auditable path from each released span to its reviewed source, redaction basis and approval.
9. Return only artifacts whose scope, sanitization, render, and recipient binding all pass. Keep the remaining document IDs withheld and challenges unresolved; the host dispatches the exact approved artifact separately.
