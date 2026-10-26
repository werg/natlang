---
description: Assemble the complete brand review deliverable from creative proposals, rule checks and revision audit.
args:{directions:BrandDirection[];system:BrandSystemState;assets:BrandAssetDraft[];constraints:{supportedClaimIds:string[];blockedClaims:string[];requiredDisclosures:string[];rightsQuestions:string[];};systemReview:{usableRules:string[];contradictions:string[];missingContexts:string[];};audit:BrandCoherenceAudit}
returns:BrandReviewPacket
---
Return the exact distinct directions, current system, adapted channel assets, all audit defects, union of audit clearance tasks and constraint/direction rights questions, and the audit's reconciled checkpoint. Preserve every prior invalidated asset and approved scope exactly as the audit returned them.

Add a blocked voice defect for each contradiction between system rules and sourced product constraints; add a review defect for each missing support or error context, attached to the affected direction or system identity. Do not discard systemReview because it is not a field in the outward packet; represent each unresolved check in defects and keep the exact finding visible in its repair text.

Set blocked when any hard claim, disclosure, rights or revision defect blocks use of an asset. Set review while a rights question, missing context, originality judgment, unapproved direction/system, invalidated asset or required client approval scope remains. Set ready only when every asset and rule is current, supported and in scope, no clearance tasks remain, the client has approved the exact current system revision, and every required asset scope is approved. Return every `BrandReviewPacket` field explicitly; no branch returns only defects or the audit wrapper.
