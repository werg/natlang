---
args: { concept: CollaborativeProjectConcept, project: ProjectDirectorRequest, prior?: ProjectDirectorCheckpoint }
returns: MilestoneCalendar
---
Divide deliverables at points where learners can inspect evidence and revise. Bind dependencies from required prior knowledge and material availability. For each learner, total planned workload against the actual available time; show critical bottlenecks separately from group work. When a milestone is missed, redistribute logistics only after identifying the learner-specific evidence that still must be produced.
Bind load = compareLearnerWorkload(milestones, project.availableWindow). When a milestone is missed bind recovery = redistributeLogisticsPreservingEvidence(milestones, prior, project.sourceEvidence); do not mark the absent learner's required demonstration complete.
