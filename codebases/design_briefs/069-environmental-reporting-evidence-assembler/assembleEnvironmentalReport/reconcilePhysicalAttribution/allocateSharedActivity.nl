---
description: Apply the approved shared-activity rule and exact unit conversion to reconciled event identities.
args: { boundary: ReportingBoundary; measures: EnvironmentalMeasure[]; identities: ActivityIdentity[] }
returns: ReconciliationLedger
---
Allocate only activity with a resolved event identity, included operation, known period, compatible unit and explicit allocation basis. Convert quantities with exact arithmetic from sourced conversion factors. Verify allocated portions conserve the source amount. Exclude unresolved events from numeric totals while retaining their source IDs and impact as gaps.
