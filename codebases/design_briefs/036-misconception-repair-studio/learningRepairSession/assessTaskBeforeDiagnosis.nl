---
args: { input: LearnerSessionInput }
returns: TaskQualityReview[]
---
1. Bind language = inspectTaskWording(input). Identify ambiguity that admits materially different interpretations.
2. Bind answerSpace = inspectAcceptedSolutionSpace(input, language). Preserve all independently valid methods, including unconventional solutions.
3. Bind prerequisites = compareTaskPrerequisites(input, answerSpace). If the task assumes a concept the learner has not been taught, mark it unsuitable for diagnosis.
4. Return task-level clarity, access barriers, competency retained, exact sources, and instructor repair questions. This review runs before learner diagnosis.
