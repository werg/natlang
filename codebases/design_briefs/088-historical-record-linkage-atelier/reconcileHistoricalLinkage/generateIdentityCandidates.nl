---
description: Generate candidate identity edges from temporal, relational, geographic and attribute evidence.
args: { dossier: LinkageDossier; change: LinkageChange; observations: HistoricalObservation[]; dates: DateInterval[]; places: HistoricalPlace[] }
returns: HistoricalCandidate[]
---
Compare pairwise records only when date intervals can plausibly overlap or an explicit relation connects them. Use kinship and relationship evidence to distinguish family members who reuse a name. A place candidate is compatible only for its effective period. Preserve both supporting and incompatible evidence on each candidate edge; do not merge candidates based on name alone.
