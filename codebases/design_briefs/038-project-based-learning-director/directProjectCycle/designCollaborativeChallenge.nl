---
args: { goals: LearningEvidenceMap, project: ProjectDirectorRequest }
returns: CollaborativeProjectConcept
---
Assign each learner role a decision or artifact that contributes to the shared challenge and produces attributable evidence for at least one goal. Check whether any role can be omitted without changing the group's result; if so, redesign that role. For unavailable materials, construct candidate substitutions from the supplied inventory and compare each against the same goal demonstration, safety constraints, and accessibility conditions.
Bind roles = makeEachRoleConsequential(goals, project.learnerGroup). Bind replacements = compareGoalPreservingMaterials(project.resourceInventory, goals, project.sourceEvidence). Return only replacements that preserve the same demonstrated skill and safe use.
