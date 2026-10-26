---
args: { location: FacilityLocationResolution, observation: FacilityObservation, history: FacilityRepairHistory[], reports: OccupantReport[] }
returns: FacilityIssueCluster
---
Link reports only by the same resolved location, compatible mechanism evidence, and supported recurrence. Keep reporter identity partitioned from shared issue state; do not equate several reports with independent faults when they describe one mechanism.
