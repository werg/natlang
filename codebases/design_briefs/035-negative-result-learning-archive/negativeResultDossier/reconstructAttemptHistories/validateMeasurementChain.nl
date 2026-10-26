---
args: { events: AttemptTimeline[], evidence: EvidencePage[] }
returns: AttemptTimeline[]
---
Trace the measurement from instrument state through calibration, acquisition, data loading, transformation, and reported result. Identify the first broken link and retain downstream values as observations that cannot support the planned hypothesis test.
