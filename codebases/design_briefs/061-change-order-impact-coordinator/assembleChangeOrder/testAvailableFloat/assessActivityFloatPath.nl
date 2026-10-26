---
args: { schedule: ScheduleActivity[], dependencyPackageIds: ID[], activity: ScheduleActivity }
returns: ActivityFloatFinding
---
1. Mark the activity affected only when activity.affectedByChange or a supplied dependency package links it to the changed scope; proximity in the schedule is not enough.
2. Traverse only predecessorIds that exist in schedule and retain the exact closure. Every missing predecessor ID is a predecessorGap; do not assume its duration or float.
3. Compare the supplied remainingDuration against availableFloat only when both use a comparable unit and the predecessor path is complete. Return "absorbed" when float covers the duration, "consumed" when positive float is reduced, "delay-risk" when duration exceeds float, and "unknown" when either value or unit basis is missing.
4. Return actual remainingFloat only when the supplied basis supports that subtraction; otherwise use an explicit unknown marker and name the missing schedule evidence in findings.
