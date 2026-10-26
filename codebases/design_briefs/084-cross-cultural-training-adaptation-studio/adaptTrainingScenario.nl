---
description: Adapt a training scenario to supplied audience context while preserving its competency and decision complexity.
args: { dossier: LearningDossier; change: AdaptationChange; reviewer: AdaptationReviewer }
returns: AdaptationProposal
---
Treat the competency and required decision as invariant. Change only incidental elements that supplied evidence shows do not fit; never generalize about an audience beyond the evidence.

Bind kernels = `recoverCompetencyKernel(dossier, change)` to separate the required decision and complexity from analogy, setting and incidental cues. Bind fit = `assessAudienceContext(dossier, change, kernels)`; preserve reviewer disagreement and narrow the scope where assumptions conflict.

Bind scenarios = `generateContextualAlternatives(dossier, change, kernels, fit)`. Replace an unfamiliar analogy without weakening the competency. Revise stereotype-risk examples only from supplied context and keep disputed assumptions visible.

Bind tests = `replayLearningDemand(dossier, change, kernels, scenarios)`. Compare adapted decision steps with the original. If the new scenario is easier, restore missing ambiguity, tradeoff or consequence while keeping unsuitable analogies removed.

Bind release = `versionTrainingRelease(dossier, change, tests, scenarios)`. Curriculum changes invalidate only adaptations whose competency mechanism changed. Preserve prior versions and require reviewer approval before release.

Return `AdaptationProposal` for `reviewer.id` with evidence-backed context fit, preserved decision demands, learner-test results, invalidations and resumable state. This is a review proposal, not a published training release.
