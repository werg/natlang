---
description: Resolve a meter replacement with source-backed calibration and continuity assumptions.
args: { dossier: ReportingDossier; change: ReportChange }
returns: string[]
---
Keep pre-replacement and post-replacement readings in separate source intervals. Require calibration evidence, effective replacement time, and the approved continuity method. If the method or calibration is missing, mark the gap and prevent a continuous total from being presented as a direct measurement.
