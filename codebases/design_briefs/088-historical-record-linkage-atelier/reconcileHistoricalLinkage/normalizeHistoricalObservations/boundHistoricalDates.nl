---
description: Convert approximate historical dates into bounded intervals while preserving the source expression.
args: { dossier: LinkageDossier; change: LinkageChange }
returns: DateInterval[]
---
Use explicit date bounds, calendar system and precision from the source or supplied chronology evidence. Preserve the raw wording. A year-only date remains a year interval; circa dates remain an interval. If calendar or bound interpretation is uncertain, retain the candidate intervals and mark unresolved rather than inventing a date.
