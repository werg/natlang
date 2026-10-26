---
args: { milestones: Milestone[] }
returns: MilestoneOrder
---
Start with milestones whose prerequisite IDs are all satisfied. Add the earliest feasible milestone to the ordered prefix, mark its deliverables satisfied, and reconsider the remaining frontier. If the frontier becomes empty while work remains, return the exact cycle or missing prerequisite set; never force an arbitrary order through a dependency contradiction.
