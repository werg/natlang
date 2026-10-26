---
description: Bind each reported measure to an operation, period, unit, method, calibration and evidence.
args: { dossier: ReportingDossier; change: ReportChange; boundary: ReportingBoundary }
returns: EnvironmentalMeasure[]
---
For every activity within the boundary, preserve the original reading and its source. Separate estimates from direct measurement and retain method, uncertainty, calibration and period. Call `resolveMeterContinuity(dossier, change)` for replaced meters; do not infer continuity from adjacent values. Leave contractor measurements unresolved until their physical activity identity is reconciled.
