---
args: { plan: AssessmentEditionPlan, receipt: AssessmentReleaseReceipt, authority: AssessmentReleaseAuthority }
returns: EditionFreezeReadiness
---
1. Require action "freeze" to match an assessment edition plan with releaseStatus and releaseDecision.status both "ready", no blockingItemIds or moderatorQuestions, and no unanswered challenge or answerability defect. Preserve the item, rubric, and challenge evidence as the frozen edition basis.
2. If any regrade has approvalNeeded true, require that regrade.applied is true before freezing the new scoring edition. A proposed score correction is not an applied regrade.
3. Require receipt.assessmentId to match both plan.caseId and authority.assessmentId; require authority.moderatorId to match receipt.moderatorId, authority to be "verified", "freeze" to appear in allowedActions, and actual evidence references from both authority and receipt.
4. Return ready only when every release, moderation, regrade, identity, and authority condition is satisfied; otherwise state the exact blockers. Do not treat a release proposal as authorization.
