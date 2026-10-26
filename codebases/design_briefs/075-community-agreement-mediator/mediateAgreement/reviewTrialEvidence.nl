---
description: Review participant-specific outcomes and enact only the agreed withdrawal exit process.
args: { dossier: MediationDossier; change: MediationChange; trial: TrialTerms }
returns: DivergentTrialFinding[]
---
Call `compareTrialOutcomes(dossier, change, trial)` and retain each participant's observations separately. Do not average divergent reports or label a trial universally successful. Apply the exact exit process when a participant withdraws, stop relying on their consent immediately, and preserve the withdrawal source and affected terms.
