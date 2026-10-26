---
args: { translations: WarningTranslation[] }
returns: WarningIntegrity
---
For each warning compare source and rendered language for every hazard, condition, and required response. Record omitted hazards and conditions by warning and target step using mapTranslationCoverage(translations). A shorter sentence is acceptable only if operational meaning is preserved; any omitted hazard blocks its step pending qualified translation review.
