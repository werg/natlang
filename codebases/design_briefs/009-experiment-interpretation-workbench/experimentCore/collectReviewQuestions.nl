---
args: { ownerId: string, timelines: EffectiveDesignTimeline[], measurements: MetricMeasurementAudit[], checks: EstimandCheck[], followups: string[] }
returns: Question[]
---
Turn only unresolved design timing, invalid or limited measurement, approval/estimand mismatch, incomplete search coverage, and missing follow-up prerequisites into questions. Route each to the supplied decision owner and name the evidence or authorization needed to resolve it. Do not emit a generic question when the child assessment contains no blocker.
