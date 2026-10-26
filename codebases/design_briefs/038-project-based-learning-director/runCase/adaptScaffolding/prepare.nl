---
args: { input: CaseInput, goal: GoalContract, policy: PolicySelection, evidencePlan: EvidencePlan, previous: ProjectMilestone[] }
returns: PreparedEvidence
---
Prepare a bounded, source-versioned evidence slice for adaptScaffolding. Interpret progress artifacts to identify where support is needed without doing the substantive work for learners. Provide targeted examples, prompts, or task decomposition, and remove support when evidence shows independent capability. Include direct observations, current authority, contrary evidence, a plausible rival interpretation, and missing decisive fields. Carry forward the actual prior-stage records when provided.
