---
description: Resolve physical activity identity before allocating measured quantities across reporting entities.
args: { dossier: ReportingDossier; boundary: ReportingBoundary; measures: EnvironmentalMeasure[] }
returns: ReconciliationLedger
---
Call `linkPhysicalActivities(dossier, measures)` and distinguish duplicate descriptions of one event from distinct activity. Then call `allocateSharedActivity(boundary, measures, identities)` only for identities supported by evidence and a stated allocation rule. Exact totals use exact unit conversion and arithmetic. Do not balance an unexplained difference or count a transferred responsibility twice.
