---
args: { scaffolds: ScaffoldPlan, calendar: MilestoneCalendar, project: ProjectDirectorRequest }
returns: ComplicationDecision
---
Recalculate remaining prerequisites, materials, time, and goal evidence after the proposed disruption. Keep a mandatory goal only when its demonstration remains feasible through an allowed route. Offer remaining routes with their tradeoffs; otherwise mark release blocked and ask the teacher to change the deadline or project scope explicitly.
Bind deadline = testRemainingDeadlineCapacity(calendar, project.availableWindow). Bind creativeRoutes = judgeCreativeRouteByCompetency(project.sourceEvidence, project.requiredGoalIds). A changed route is acceptable only when its evidence proves the same required skill and does not consume the remaining time needed for other mandatory goals.
