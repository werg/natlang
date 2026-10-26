---
args: { input: LearnerSessionInput }
returns: TaskQualityReview[]
---
Check whether the prompt, notation, reference frame, and requested response form have one intended interpretation. When multiple reasonable readings produce different responses, record the interpretations and ask for task repair before interpreting a student answer.
