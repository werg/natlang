---
description: Preserve source record wording and map dates and places to uncertainty-aware historical representations.
args: { dossier: LinkageDossier; change: LinkageChange }
returns: HistoricalNormalization
---
For every record, preserve the recorded name, raw date and place expressions, source version and attributes. Bind dates = `boundHistoricalDates(dossier, change)` to represent approximate dates as intervals. Bind places = `mapHistoricalPlaces(dossier, change)` to map place expressions only to boundaries supported for the relevant period. Retain multiple jurisdiction candidates where boundary evidence is unresolved. Return observations, dates, and places together as HistoricalNormalization.
