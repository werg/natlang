---
args: { prior: QualitativeStudyCheckpoint | null, proposal: QualitativeStudyProposal }
returns: StudyRevisionImpact
---
Compare corpus version, selected and excluded sample IDs, and each code's versioned inclusion and exclusion boundary with the saved checkpoint. Mark coding stale only when its passage, sample membership or applied code boundary changed. A privacy event blocks quotation export but does not erase the restricted analytic record. With no checkpoint, report no previously valid coding as stale and surface all consent-uncertain passages for review.
