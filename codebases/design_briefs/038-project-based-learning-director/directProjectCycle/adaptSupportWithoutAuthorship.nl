---
args: { calendar: MilestoneCalendar, project: ProjectDirectorRequest, prior?: ProjectDirectorCheckpoint }
returns: ScaffoldPlan
---
Locate the exact milestone where progress evidence shows the learner is stuck. Offer prompts, accessible representations, or a smaller practice step tied to that obstacle; never produce the final assessed artifact. If group authorship is uncertain, hold the group artifact as shared evidence only and request short individual demonstrations against the same competency rubric.
Bind substitution = inspectSupportForAuthorshipLeak(calendar, prior, project.sourceEvidence). Bind demonstrations = designIndividualAuthorshipChecks(substitution, project.requiredGoalIds). Remove any generated completion text from the learner evidence path.
