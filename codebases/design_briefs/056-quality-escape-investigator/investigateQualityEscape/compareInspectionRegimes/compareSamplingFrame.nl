---
args: { defects: DefectObservation[], methods: InspectionMethod[], evidence: Evidence[] }
returns: SampleFrameReview
---
Recover each inspection denominator, sampled population, season or production period, and method revision from supplied evidence. Pair observations only when the compared populations and windows align; expose missing denominator and method changes separately from defect counts.
