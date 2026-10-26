---
description: Test each proposed narrative claim against the actual measured outcome and boundary scope.
args: { dossier: ReportingDossier; measures: EnvironmentalMeasure[]; reconciliation: ReconciliationLedger; change: ReportChange }
returns: ClaimAssessment[]
---
For every claim, follow its measure and activity references to current source records. Check whether the statement refers to an activity completed, a measured outcome, or an estimated effect; these are different assertions. A completed initiative alone cannot substantiate benefit. Qualify estimates with their method and limitations. Mark claims unsupported when their scope exceeds the measure or the evidence is outside the approved boundary.
