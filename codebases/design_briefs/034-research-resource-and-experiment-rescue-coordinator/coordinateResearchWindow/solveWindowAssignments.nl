---
args: { mandates: ProtocolMandate[], resources: ResourceAssessment[], snapshot: ResearchOperationsSnapshot }
returns: AssignmentSearch
---
1. Bind seed = seedAssignmentSearch(mandates, resources). The frontier retains every requested experiment and its still-possible resource/time branches.
2. Bind explored = walkScheduleFrontier([seed], mandates, resources, snapshot). NatLang advances every branch one experiment at a time through a bounded breadth-first frontier, retaining rejected reasons; do not hide policy or branch choice in crisp code.
3. Bind selected = chooseBestProtocolSchedule(explored, mandates). First maximize hard-condition satisfaction, then compare explicit project priority and dependency order among feasible alternatives. Preferences cannot admit an infeasible branch.
4. Return assignments, unscheduled IDs, exact conflicts, search-completion state, and branch count. Never relax protocol predicates to make a schedule appear feasible.
