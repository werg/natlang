---
description: Recover prior terms and participant-specific acceptance without resolving disputed recollection by majority.
args: { dossier: MediationDossier; change: MediationChange }
returns: ParticipantTermState[]
---
Call `compareAgreementAccounts(dossier, change)` for each disputed term. Preserve the prior accepted version and every competing account with its source. Mark each participant term accepted, declined, pending, or withdrawn from explicit current evidence. A nonresponse remains pending. Do not rewrite a prior agreement from recollection alone.
