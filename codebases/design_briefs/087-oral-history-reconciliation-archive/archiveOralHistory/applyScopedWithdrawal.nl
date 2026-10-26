---
description: Apply a speaker's withdrawal only to the anecdote and derivatives within its consent span.
args: { dossier: OralHistoryDossier; change: ArchiveChange; accounts: OralHistoryAccount[] }
returns: DerivativeLineage[]
---
Call `invalidateSpanDerivatives(dossier, change, accounts)` using the exact withdrawn source span and its derivative lineage. Stop further use of the anecdote. Retract its quotations and summaries, but retain unrelated material when its own consent remains current. Record each affected derivative and notification obligation without deleting the original audit history.
