---
description: Separate visible condition, reported function and item identity evidence.
args:{conditionReports:EvidencePage[];imageAnnotations:EvidencePage[];serialLedger:EvidencePage[]}
returns:ConditionEvidenceView
---
Record cosmetic observations only from inspection or structured image annotations; functional symptoms remain attributed to reporter or test. List every serial found and whether it matches the original order. Preserve conflicting observations and exact evidence references. Return missing decisive evidence rather than a single unsupported grade.
