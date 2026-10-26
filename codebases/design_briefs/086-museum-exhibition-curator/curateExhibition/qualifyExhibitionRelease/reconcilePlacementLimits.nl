---
description: Reconcile each object's planned placement exposure and duration with its current conservation limits.
args: { dossier: ExhibitionDossier; change: ExhibitionChange; constraints: ObjectConstraint[] }
returns: string[]
---
For each object placement, compare the actual planned lux and duration with the current conservation report. Do not substitute a prior limit after a condition update. Identify affected placements, end dates and evidence. If the plan does not provide actual exposure or duration, block that placement pending an exact facilities plan.
