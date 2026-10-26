---
args: { observation: FacilityObservation, patterns: ComplaintTimePattern[], issues: FacilityIssueCluster, visit: ConsentedVisitPlan }
returns: FacilityObservationGate
---
Recover the actual occurrence pattern from observation and time-pattern evidence for issues.issueIds. Keep incompatible windows or disputed conditions separate. Select only observations whose relevant conditions intersect visit.windows and whose access is currently consented. A convenient visit outside the reported condition cannot verify absence of the complaint.

Bind observationPlan to the symptom, agreed window, evidenced condition and objective check needed to distinguish the supported mechanisms. If no relevant pattern is known, request an observation diary or a permitted diagnostic visit without promising resolution. If windows do not overlap, request a new consented opportunity; never widen consent. Return FacilityObservationGate { observationPlan, blockers: the exact unknown condition, non-overlap or consent gaps }. An observation plan supplies no actual technician result.
