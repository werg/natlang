---
description: Compare adapted scenarios against original learning demands and preserve decision complexity.
args: { dossier: LearningDossier; change: AdaptationChange; kernels: CompetencyKernel[]; scenarios: AdaptedScenario[] }
returns: ComplexityComparison[]
---
Call `compareRequiredDecisionSteps(kernels, scenarios)` for each candidate. Preserve the same essential choice, competing considerations, consequence and learner action. If the adapted scenario removes a relevant decision demand, revise it or block it; do not call an easier scenario equivalent merely because learners finish it.
