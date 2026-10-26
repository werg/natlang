---
args: { dossier: EngineeringChangeDossier, plan: EngineeringChangePlan, book: RetrofitExecutionBook, receipt: RetrofitExecutionReceipt, approvedStepIds: ID[], targetApprovedRevision: string }
returns: RetrofitExecutionAssessment
---
1. Bind receipt.changeId to plan.changeId, the unit to dossier.fieldedUnits, and receipt.fromConfiguration to that unit's current configuration in book. A stale unit baseline cannot advance.
2. Require receipt.targetConfiguration to be an affected approved configuration for this change. For a completed step, require stepId in approvedStepIds, the installedRevision equal targetApprovedRevision, and verificationRefs proving the named step passed.
3. Keep the unit mixed until receipts for every required step in approvedStepIds are verified for this unit and configuration. Never mark the unit complete merely because the final-install operation ran.
4. For a reversed step, require reversalRefs and evidence that the unit returned to its actual fromConfiguration. Unknown installation or verification status keeps the configuration unknown and fenced.
5. Return target, mixed, baseline, or unknown status with every missing verification and revision mismatch.
