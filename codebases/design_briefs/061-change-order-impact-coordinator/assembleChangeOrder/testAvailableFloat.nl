---
args: { schedule: ScheduleActivity[], dependencies: ID[] }
returns: ScheduleImpact
---
Bind findings = schedule.map(activity => assessActivityFloatPath(schedule, dependencies, activity)). Use predecessorClosure and predecessorGaps to distinguish a supported path from a disconnected duration. Bind affectedActivityIds from findings on affected paths and blockers from predecessorGaps and unknown float outcomes. Return no point-date claim when findings contain missing critical-path or duration inputs.
