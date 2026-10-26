---
description: Distinguish confirmed resource failure from uncertain restoration estimate.
args:{event:EventProductionState;reports:Disruption[]}
returns:Disruption[]
---
Require each report to identify a capability, affected segments, report time and source. Keep estimated loss separate from confirmed failure and do not infer a convenient return time. Reconcile restoration only from current authoritative telemetry; preserve disagreement for producer decision.
