---
args: { rubrics: RubricSpec[], challenges: ChallengeAdjudication[], assessment: AssessmentBuildRequest }
returns: RegradePlan[]
---
For each changed rubric version identify the full administration cohort and all response identifiers scored under the prior version. Derive proposed score changes from the repaired criterion and preserved response evidence, including learners who did not appeal. Leave a change pending until an authorized moderator approves this exact cohort, old/new version pair, and correction rule. Do not write grades.
