---
args: { schedule: ScheduleActivity[], dependencies: ID[] }
returns: ScheduleImpact
---
Use exact supplied predecessor links, remaining durations, and available float to identify activities where the change consumes float and those where float absorbs it. If critical-path or duration inputs are missing, return assumptions and a blocker instead of a point date claim.
