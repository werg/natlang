---
args: { request: DisclosureReviewRequest, artifacts: ReleaseArtifact[], approvals: RecipientApproval[] }
returns: RecipientApprovalReview
---
Initialize authorizedArtifactHashes, approvalRefs and blockers as empty arrays. For each actual artifact require its recipientId and purpose to equal request's recipient and purpose, then inspect approvals for that exact recipient, purpose and artifact.renderedHash. Require a nonempty approval reference and version; keep conflicting approval rows blocked. This input contains no revocation service; the caller must supply the current authorized approval set. Approval for sanitized input bytes does not authorize a different rendered output.

Append only the approved artifact's renderedHash and the matching approvalRef. For every unmatched artifact record a blocker naming its document and exact missing or mismatched approval binding. Do not infer permission from an approval belonging to another document, recipient, purpose or output hash. Return RecipientApprovalReview { authorizedArtifactHashes, approvalRefs, blockers }, including complete empty/blocked fields when no artifacts exist.
