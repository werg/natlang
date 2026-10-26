---
args: { items: AssessmentItemDraft[], answerability: AnswerabilityLedger, rubrics: RubricSpec[], challenges: ChallengeAdjudication[], repairs: RepairRegressionReport, assessment: AssessmentBuildRequest }
returns: EditionReleaseDecision
---
Block release if any item lacks a supported answer path, a material reading changes the answer, a model answer is wrong, a challenge remains unresolved, or a repair regression is open. A clean blueprint alone cannot override a correctness defect. When a key error affects an already administered edition, keep release blocked and surface the linked controlled regrade plan. Return the specific item IDs and unresolved moderator questions; otherwise mark ready for accountable approval, not published.
