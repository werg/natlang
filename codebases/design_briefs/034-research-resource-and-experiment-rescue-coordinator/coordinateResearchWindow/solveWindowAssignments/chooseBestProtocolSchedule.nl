---
args: { explored: AssignmentFrontier[], mandates: ProtocolMandate[] }
returns: AssignmentSearch
---
Compare all complete frontiers first by satisfaction of protocol hard predicates. Among equally valid schedules, apply the documented priority and dependency ordering; then preserve preference tradeoffs. Build `unscheduledExperimentIds` from the chosen frontier's deferredExperimentIds together with any IDs still pending because the finite search was incomplete, and retain rejected hard predicates from the chosen frontier. Do not report an intentionally deferred feasible assignment as a hard failure. If search is incomplete, do not claim an optimal schedule.
