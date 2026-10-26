---
description: Create an immutable submission edition and explicit restatement bridge from the reconciled report ledger.
args: { dossier: ReportingDossier; boundary: ReportingBoundary; reconciliation: ReconciliationLedger; claimAssessments: ClaimAssessment[]; change: ReportChange }
returns: EnvironmentalSubmission
---
Call `deriveRestatementBridge(dossier, boundary, reconciliation)` when boundary, meter method, estimate status, or source value changed. Keep the prior edition immutable and attribute each changed total to a scope, method, measurement, or operational effect. Include unresolved gaps, claim limitations, reviewer, required schedules and approval references. Do not mark the report submitted or approved without a corresponding authority receipt.
