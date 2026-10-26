---
description: Bind brand promises and prohibitions to current product evidence.
args:{brief:ProductBrief;claimEvidence:ClaimEvidence[];sourceRecord:BrandProject;sourceEvidence:EvidenceItem[]}
returns:{supportedClaimIds:string[];blockedClaims:string[];requiredDisclosures:string[];rightsQuestions:string[];}
---
For the controlling BrandProject, use its top-level clientId, creativeBrief, productFactIds, audienceGuidanceId and creativeDirectorId as the source of product scope, audience and accountable review. Match productFactIds to current sourceEvidence and claimEvidence. Any capability claim unsupported for this client and revision stays blocked; do not trust the supplied generic brief to supersede the project record.

Match every proposed capability claim to current supported evidence for this exact product revision. Keep disputed or withdrawn claims blocked. Carry prohibitions and required disclosures from the brief. Identify names or visual elements requiring legal/rightsholder clearance as questions, not conclusions.
