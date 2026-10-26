---
args: { portfolio: PortfolioAssessmentRequest }
returns: PortfolioAssessmentProposal
---
1. Bind indexed = distinguishPortfolioContributions(portfolio). Keep direct artifacts, learner reflection, group products, and supervisor observations as different evidence classes.
2. Bind mappings = mapObservedWorkToCompetencies(indexed, portfolio). Require a source location and a demonstrated behavior for each claimed competency; keyword overlap is insufficient.
3. Bind findings = testEvidenceSufficiency(mappings, portfolio). Apply the supplied framework standard and recency rule; check independence, breadth, and contradictory evidence without averaging away conflict.
4. Bind tasks = closeMostConsequentialEvidenceGap(findings, portfolio). Define the acceptance evidence before asking for new work. If an accommodation changes the format, retain the competency contract and use only an approved alternate demonstration.
5. Bind attributionAppeal = rehearAttributionCorrection(portfolio.priorFinding, portfolio). Reassess only affected competencies while preserving the old finding and evidence versions.
6. Bind disposition = assembleAssessorDecision(indexed, mappings, findings, tasks, attributionAppeal, portfolio). Return { caseId: portfolio.caseId, indexed, mappings, findings, tasks, attributionAppeal, disposition, approvalRequired: portfolio.stakes !== 'ordinary', commitAllowed: false }.
