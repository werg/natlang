---
description: Produce exact witnesses for jointly unsatisfiable hard positions.
args:{positions:StakeholderPosition[];constraints:EvidencePage[]}
returns:ConflictWitness[]
---
For each relevant pair compare its required values with explicit resource or interface limits in the supplied constraints. State the precise witness and minimal relaxation that would restore compatibility. Return no hard-conflict witness from absence of evidence.
