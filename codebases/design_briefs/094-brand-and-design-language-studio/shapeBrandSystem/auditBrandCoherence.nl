---
description: Review hard consistency defects and scoped approvals across brand derivatives.
args:{brief:ProductBrief;system:BrandSystemState;assets:BrandAssetDraft[];constraints:{supportedClaimIds:string[];blockedClaims:string[];requiredDisclosures:string[];rightsQuestions:string[];}}
returns:BrandReviewPacket
---
Trace each asset claim to current evidence and each asset voice choice to a system rule. Identify inconsistent product terms and missing disclosures as hard defects. Keep originality and subjective preference for human review. Confirm client approval scope against the exact revision; a narrow approval never expands by implication. Invalidate only descendants of changed facts or system rules and return the complete dependency checkpoint.
