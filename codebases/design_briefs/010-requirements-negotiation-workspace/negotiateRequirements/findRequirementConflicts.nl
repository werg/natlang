---
description: Test stakeholder positions pairwise against explicit technical, time and capacity constraints.
args:{scope:GoalContract;positions:StageResultStakeholderPosition;constraints:EvidencePage[]}
returns:StageResultRequirementConflict
---
Compare only position pairs that can affect the same release, resource or outcome. Bind witnesses = `compareHardConstraintPairs(positions.accepted, constraints)` and terminology = `testRequirementTerminology(positions.accepted, constraints)`. A hard conflict requires an exact witness; a semantic conflict retains explanatory counterexamples. If no governing limit is evidenced, leave the conflict unresolved as a question instead of claiming infeasibility. Return typed conflict records plus complete coverage accounting.
