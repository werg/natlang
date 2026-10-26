---
args: { analysis: SemanticWorkOrderRequest }
returns: AnalysisMeaningMap
---
Bind population, entity, semantic predicates, time relation, grouping, and requested result grain from the question. Compare alternate readings only when they change which records or aggregates qualify. If “after” lacks a window or reference event, keep the alternatives and ask the requester before selecting any available timestamp.
