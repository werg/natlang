---
args: { frontier: AssignmentFrontier, mandates: ProtocolMandate[], resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot }
returns: AssignmentFrontier[]
---
1. Bind id = chooseNextExperiment(frontier.pendingExperimentIds, mandates, resources).
2. Bind options = deriveLegalIntervalOptions(id, mandates, resources, snapshot, frontier.selected).
3. Bind checked = testAssignmentFeasibility(options.branches, mandates, resources, snapshot, frontier.selected).
4. If checked.feasible is empty, return one frontier with id removed from pending and its exact blocked reasons retained. Otherwise return one child frontier per feasible branch, each with only that branch's assignment appended and the selected experiment removed from pending.
