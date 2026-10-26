---
args: { facts: ReuseSiteFact[], drawings: HistoricPlan[], surveys: CurrentSurvey[] }
returns: SiteEvidenceReconciliation
---
Keep historic drawing claims and current survey observations as separate versioned facts. Prefer a qualified recent survey for observed present condition only within its measured area and method; report conflicts where coverage or qualification does not support a resolution.
