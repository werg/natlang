---
args: { issues: FacilityIssueCluster, visit: ConsentedVisitPlan, history: FacilityRepairHistory[] }
returns: FacilityWorkOrderPlan
---
Create work tied to the linked mechanism and a visit observation under the relevant conditions. When prior work recurs without verified resolution, escalate mechanism investigation and retain the prior work-order link rather than opening an unrelated duplicate.
