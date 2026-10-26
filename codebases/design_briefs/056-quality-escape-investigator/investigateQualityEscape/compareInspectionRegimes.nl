---
args: { defects: DefectObservation[], methods: InspectionMethod[], evidence: Evidence[] }
returns: DefectRateComparison
---
Bind methods = compareMethodValidityWindows(methods). Bind frame = compareSamplingFrame(defects, methods, evidence). A raw rate is comparable only when both method validity and sampling denominators, population, and time window align. Preserve individual defect observations even when an aggregate comparison is invalid.
