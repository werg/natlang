---
args: { request: GrantCycleRequest; narrative: GrantNarrativeDraft; evidence: EvidencePage[]; correction: GrantCorrectionPlan }
returns: GrantSubmissionPlan
---
Check required forms and schedules against the exact funder package evidence. Bind the local artifact version and approval reference to its exact content hash before preparing dispatch. If this is a correction, link the new version to the old submission and receipt; never replace that record. Leave submittedAt and receiptRef empty unless a supplied receipt proves actual portal acceptance. Return a dispatch-ready plan only after required evidence, approval and artifact consistency are established.