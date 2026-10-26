---
description: Identify apparent conflicts caused by terms whose meanings differ across stakeholders.
args:{positions:StakeholderPosition[];constraints:EvidencePage[]}
returns:ConflictWitness[]
---
Compare defined terms and examples in source context. Preserve each interpretation and give a concrete counterexample that shows whether behavior truly differs. Do not collapse vocabulary differences into hard capacity constraints.
