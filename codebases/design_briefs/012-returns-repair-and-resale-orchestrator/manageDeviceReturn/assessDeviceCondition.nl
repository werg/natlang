---
description: Reconcile customer, inspection and image evidence without unsupported diagnosis.
args:{scope:GoalContract;policy:PolicySelection;conditionReports:EvidencePage[];imageAnnotations:EvidencePage[];serialLedger:EvidencePage[]}
returns:StageResultConditionAssessment
---
Bind observations = `separateConditionObservations(conditionReports, imageAnnotations, serialLedger)`. Keep cosmetic findings, functional symptoms and allegations separate; bind each to the exact item and evidence source. Do not infer physical findings from an unavailable image. Require matching serial evidence to the original order before associating an inspection. If symptoms are intermittent, keep diagnosis unknown and ask for the specific routing-changing test.
