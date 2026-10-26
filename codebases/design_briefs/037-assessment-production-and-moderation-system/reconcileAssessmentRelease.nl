---
args: { ledger: AssessmentEditionLedger, plan: AssessmentEditionPlan, receipt: AssessmentReleaseReceipt, authority: AssessmentReleaseAuthority }
returns: EditionReleaseReconciliation
---
1. For a consumed receiptId, return the unchanged ledger as "replayed" only when the stored receipt matches every field. Reused identity with changed contents is held and never overwrites history.
2. For a "freeze" receipt, bind readiness = verifyEditionFreezeReadiness(plan, receipt, authority). Also require receipt.editionId and editionVersion to be present and planDigest to bind the exact compiled edition. If readiness is false, preserve the receipt as held evidence and do not add a frozen edition.
3. For a "retire" receipt, locate the exact existing edition and require its assessmentId to match, authority.verified to be true, receipt.moderatorId to equal authority.moderatorId, "retire" in allowedActions, and actual evidenceRefs. Retirement prevents future use but preserves prior learner responses, scoring history, and appeal links.
4. If a new receipt tries to reuse an editionId with a different version or planDigest, preserve it as held evidence and request a fresh edition identity. Never edit a frozen edition in place.
5. Append the exact external receipt once. Only an authorized freeze or retirement changes the corresponding edition record; failed or ambiguous attempts remain as receipt evidence with blockers. Increment ledger.revision once and preserve all other editions and receipts.
6. Return frozen, retired, held, or replayed with outstanding appeal, regrade, or moderator follow-up. A frozen edition is available for administration but does not assert that a learner sat or passed it.
