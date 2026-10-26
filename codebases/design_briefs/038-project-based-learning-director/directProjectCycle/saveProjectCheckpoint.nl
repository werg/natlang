---
args: { prior?: ProjectDirectorCheckpoint, calendar: MilestoneCalendar, scaffolds: ScaffoldPlan }
returns: ProjectDirectorCheckpoint
---
Version the checkpoint from the exact previous version. Persist completed milestone identifiers, learner-specific evidence still due, scaffold exposure, and unresolved dependency questions. Do not turn a group artifact into an individual's completed evidence when resuming.
