---
description: Propose a bounded reversible trial only with exact participant terms and individual acceptance.
args: { dossier: MediationDossier; positions: MediationPosition[]; options: CommunityOption[]; change: MediationChange }
returns: TrialTerms
---
Call `qualifyTrialAcceptance(dossier, options, change)` before marking a trial ready. Specify exact participants, each tested term, start, end, review trigger and agreed exit process. Every participant must accept the same trial version explicitly. Missing or conflicting consent leaves the trial pending and prevents it from becoming an agreement.
