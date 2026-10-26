---
args: { metric: string, window: string, schemas: EvidencePage[], missingness: EvidencePage[] }
returns: MissingnessWindowReview
---
Bind exposure = traceWindowExposure(metric, window, schemas, missingness). Compare each arm's eligible population, observation opportunity and missing outcome rate using only supplied denominators. Preserve source measurement missingness separately from true outcome absence. Return whether treatment exposure changed, the observed shift, evidence identities, comparability effect and unresolved denominator questions.
