---
description: Shape a versioned brand system and channel assets from sourced product facts.
args:{ sourceRecord: BrandProject; sourceEvidence: EvidenceItem[];brief:ProductBrief;claimEvidence:ClaimEvidence[];directions:BrandDirection[];system:BrandSystemState;assets:BrandAssetDraft[];checkpoint:BrandCheckpoint}
returns:BrandReviewPacket
---
Read `clientId`, `creativeBrief`, `productFactIds`, `audienceGuidanceId` and `creativeDirectorId` from the top level of the BrandProject record. Apply those verified product claims and prohibitions to each asset branch; do not substitute a record.fields lookup. Carry every source fact and evidence ref into the analysis branch that depends on it; when any transformed proposal conflicts with this anchor, hold that branch for review instead of accepting the placeholder.

Bind constraints = `recoverProductConstraints(brief, claimEvidence, sourceRecord, sourceEvidence)`. Aspirational positioning remains distinct from demonstrated capability; unsupported claims and prohibited promises are hard constraints. Bind directions = `differentiateCreativeDirections(brief, directions, constraints)`. Keep directions materially distinct in message logic, emphasis, and exclusions; create a rights-clearance task for a potentially conflicting name rather than asserting availability.

Bind systemReview = `testBrandRules(system, directions, constraints)`. Require separate voice behavior for promotion, errors, and sensitive support interactions. Bind channelAssets = `adaptBrandAssets(brief, assets, system, constraints)`. Each asset must serve its channel's actual user task, preserve required disclosure, and cite only supported claims.

Bind defects = `auditBrandCoherence(brief, system, channelAssets, constraints)`. Product revisions invalidate only dependent claims/assets. A client approval must match its exact direction, system, or asset scope; approval of one slogan cannot approve the system. Return a review packet, and mark ready only when hard defects clear and all required approval scopes are evidenced.
