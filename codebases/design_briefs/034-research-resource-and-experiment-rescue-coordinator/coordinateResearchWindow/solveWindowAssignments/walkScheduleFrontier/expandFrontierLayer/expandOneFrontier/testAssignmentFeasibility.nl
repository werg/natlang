---
args: { branches: AssignmentBranch[], mandates: ProtocolMandate[], resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot, selected: WindowAssignment[] }
returns: FeasibilitySet
---
For every branch bind overlaps = testAssignmentConflicts(branch, selected, snapshot). Then check the exact resource capability, protocol operator and duration, calibration through slot end, and sample-window containment. Add a branch to feasible only when all hard predicates hold and overlaps is false. Record each rejected branch with the precise failed predicate; don't turn a preference into rejection of a hard-valid schedule.
