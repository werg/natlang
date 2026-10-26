---
description: Compare each participant's report of a bounded agreement trial without erasing divergent outcomes.
args: { dossier: MediationDossier; change: MediationChange; trial: TrialTerms }
returns: DivergentTrialFinding[]
---
Bind each outcome to the participant, tested term, observation period and source. Report agreement, disagreement and missing response separately. A positive result for some participants cannot override a negative or unknown result for another participant.
