---
args: { criteria: EligibilitySnapshot, lineage: StudyLineage, records: Evidence[] }
returns: ObservationLedger
---
Apply each frozen criterion to source-backed study reports. For eligible studies extract design, population, outcome, estimate, original unit, follow-up, limitations and exact evidence locator. If only an abstract is accessible, record the access limit and do not infer unreported results.
