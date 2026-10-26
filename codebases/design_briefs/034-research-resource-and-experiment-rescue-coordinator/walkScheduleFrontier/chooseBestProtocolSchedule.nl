---
args: { explored: AssignmentFrontier, mandates: ProtocolMandate[] }
returns: AssignmentSearch
---
Compare all complete frontiers first by satisfaction of protocol hard predicates. Among equally valid schedules, apply the documented priority and dependency ordering; then preserve preference tradeoffs. Keep unscheduled experiment IDs and rejected hard predicates in the result. If search is incomplete, do not claim an optimal schedule.
