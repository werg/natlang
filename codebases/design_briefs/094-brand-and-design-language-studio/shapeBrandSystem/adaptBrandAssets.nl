---
description: Adapt approved brand principles to user tasks and channel limits.
args:{brief:ProductBrief;assets:BrandAssetDraft[];system:BrandSystemState;constraints:{supportedClaimIds:string[];blockedClaims:string[];requiredDisclosures:string[];rightsQuestions:string[];}}
returns:BrandAssetDraft[]
---
For each asset, match its channel and user task to a current system rule. Preserve claim provenance and required disclosures within the channel schema. Do not treat a slogan as task information. Flag missing product facts, length or format limits rather than filling them with invented claims.
