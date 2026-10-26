---
args: { issues: FacilityIssueCluster, visit: ConsentedVisitPlan, history: FacilityRepairHistory[], observation: FacilityObservation, patterns: ComplaintTimePattern[] }
returns: FacilityWorkOrderPlan
---
Bind observationGate = selectSymptomObservation(observation, patterns, issues, visit). Build technical tasks only for the actual symptom and issue identities that can be observed under the returned conditions; a generic room inspection does not close a condition-dependent complaint. Carry observationGate.blockers and unresolved mechanism links that prevent choosing an appropriate trade into blockers.

For each supported recurring issue, retain its actual priorWorkOrder link and propose a mechanism investigation explaining what the earlier intervention failed to establish. Preserve separate tasks for different supported mechanisms. Bind escalation to recurrence, unresolved safety symptoms and evidence gaps requiring a qualified specialist; an escalated task is not already dispatched or completed. Return FacilityWorkOrderPlan { tasks, observationPlan: observationGate.observationPlan, escalation, blockers }. With no consented or relevant observation opportunity, return no executable task and keep the requested observation/consent work visible.
