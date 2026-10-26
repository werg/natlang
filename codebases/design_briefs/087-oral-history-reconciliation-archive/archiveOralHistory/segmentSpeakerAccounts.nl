---
description: Segment speaker accounts with exact source spans, attribution and remembered certainty language.
args: { dossier: OralHistoryDossier; change: ArchiveChange }
returns: OralHistoryAccount[]
---
Bind every account to interview ID, transcript version, speaker and source time span. Preserve the speaker's words indicating uncertainty, memory or correction. Do not convert a recollection into a verified fact. Keep each anecdote separately addressable so consent changes can affect only the source span it covers.
