---
args: { evidence: IndexedPortfolioEvidence[], mappings: PortfolioCompetencyMap[], findings: PortfolioCompetencyFinding[], tasks: PortfolioEvidenceTask[], appeal: AttributionAppealReview, portfolio: PortfolioAssessmentRequest }
returns: AssessorDisposition
---
Bind unsupported = identifyClaimedOutcomesWithoutEvidence(mappings, findings). Recommend only an outcome supported by the supplied standard and evidence. For high-stakes work, mark the result pending named assessor approval and expose the appeal route. Preserve assessor disagreements, missing facets, recency limits, and requested next tasks in the review packet.
