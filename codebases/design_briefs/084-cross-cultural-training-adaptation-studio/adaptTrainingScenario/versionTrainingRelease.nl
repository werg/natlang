---
description: Bind learner replay results and reviewer approval to a versioned adaptation release.
args: { dossier: LearningDossier; change: AdaptationChange; tests: AdaptationTest[]; scenarios: AdaptedScenario[] }
returns: AdaptedTrainingRelease
---
Compare every test result with the original competency criterion and preserve learner errors, not just completion rates. Bind the release to the scenario, curriculum, context evidence and reviewer decision versions. Invalidate only scenarios whose source curriculum mechanism changed. Require explicit reviewer approval before a release is considered publishable.
