---
args: { prior: PortfolioAssessmentFinding | null, portfolio: PortfolioAssessmentRequest }
returns: AttributionAppealReview
---
When prior is null, preserve the supplied prior evidence record and report that no earlier finding is available to revise. Otherwise compare the learner's correction with the source record for the disputed contribution. Preserve the earlier finding and its source versions. Identify the competencies whose evidence set changes, then request reassessment only for those competencies. Leave the original assessor decision intact until a distinct authorized assessor issues a versioned replacement.
