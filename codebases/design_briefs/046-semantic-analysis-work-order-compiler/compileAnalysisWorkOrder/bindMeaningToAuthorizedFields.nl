---
args: { meaning: AnalysisMeaningMap, analysis: SemanticWorkOrderRequest }
returns: FieldSupportMap
---
For every required concept find a data field whose documented semantics match the requested meaning. Distinguish absent fields from inaccessible records and identify proxies that change the question. Remove restricted record identifiers and values before any semantic classification request; report excluded coverage without exposing them.
Bind authorizedUniverse = excludeRestrictedRowsBeforeExposure(analysis.accessRules, analysis.requesterId). Match semantic fields only against this approved universe.
