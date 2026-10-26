---
args: { defects: DefectObservation[], methods: InspectionMethod[], evidence: Evidence[] }
returns: DefectRateComparison
---
Bind methodChanges = compareMethodValidityWindows(methods). Align each reported defect with the inspection revision and coverage active when it was observed. A different threshold or sampled population makes a raw count comparison non-comparable. Keep observed defects as evidence even when the detection regime changed.
