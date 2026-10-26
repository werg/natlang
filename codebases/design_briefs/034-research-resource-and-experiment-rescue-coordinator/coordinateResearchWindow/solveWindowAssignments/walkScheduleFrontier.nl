---
args: { frontiers: AssignmentFrontier[], mandates: ProtocolMandate[], resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot }
returns: AssignmentFrontier[]
---
Starting from the supplied frontiers, expand one pending experiment per frontier for each remaining depth through expandFrontierLayer. Stop when every frontier has no pending experiment or the finite experiment count is exhausted. Keep each alternative schedule as its own frontier, including branches that leave an experiment unscheduled; retain all rejected reasons, conflict sets, branch counts, and completeness flags.
