---
args: { assessment: AssessmentBuildRequest }
returns: CompetencyTrace[]
---
Extract each competency and its approved weight from the current blueprint. Reject a blueprint where weights conflict or a skill has no observable evidence path.
