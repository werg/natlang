---
args: { frontiers: AssignmentFrontier[], mandates: ProtocolMandate[], resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot }
returns: AssignmentFrontier[]
---
For each incomplete frontier call expandOneFrontier and append every returned feasible branch. Preserve already-complete frontiers unchanged. If a branch has no feasible assignment, retain a separate unscheduled branch with its exact blocked reason. Do not merge frontiers that differ in assignments or rejected candidates.
