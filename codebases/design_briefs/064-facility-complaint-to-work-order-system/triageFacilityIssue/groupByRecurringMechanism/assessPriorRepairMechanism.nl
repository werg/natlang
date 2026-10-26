---
args: { location: FacilityLocationResolution, observation: FacilityObservation, prior: FacilityRepairHistory }
returns: FacilityRecurrenceFinding
---
Require a resolved official location equal to prior.locationId before comparing symptoms. Compare observation.symptom and occurrencePattern with the actual prior mechanism and verifiedResult. A previous work-order number proves a recorded intervention, not that it fixed the cause. Establish recurrence only when recurrenceEvidence supports return of the compatible symptom after that intervention; an unsupported allegation stays unknown.

Return FacilityRecurrenceFinding { issueId: prior.issueId, relationship: "recurrence", "compatible", "different" or "unknown" as supported, mechanism: prior.mechanism, priorWorkOrderId: prior.priorWorkOrder, evidenceRefs: actual recurrenceEvidence and observation.sourceRefs used, blockers: unresolved location, timing, mechanism or result facts }. A compatible result without recurrence evidence can justify investigation but never a completed repair claim.
