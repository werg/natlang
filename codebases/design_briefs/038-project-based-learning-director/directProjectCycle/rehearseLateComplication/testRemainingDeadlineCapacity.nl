---
args: { calendar: MilestoneCalendar, window: string }
returns: DeadlineFeasibility
---
Remove only completed milestone work from the remaining plan. Re-evaluate each outstanding dependency, resource need, and learner workload against the actual remaining window. Report any goal whose evidence can no longer be produced before the deadline; do not claim release feasibility from group completion percentages alone.
