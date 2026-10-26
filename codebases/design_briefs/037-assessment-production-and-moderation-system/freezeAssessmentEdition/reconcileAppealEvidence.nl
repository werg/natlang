---
args: { response: LearnerResponse, rubric: RubricSpec, moderation: ModerationContext }
returns: AppealFinding
---
Bind method = semanticMethodAppeal(response.reasoning, rubric, response.item). Compare the actual response with the frozen rubric version that governed its administration. If method is a credible competency-preserving alternative, route it to an independent subject reviewer and retain the original score pending that decision. If the rubric itself omitted the method, enumerate every response scored under that rubric version before proposing any correction. Return affected response identifiers and the evidence basis; do not apply score changes here.
