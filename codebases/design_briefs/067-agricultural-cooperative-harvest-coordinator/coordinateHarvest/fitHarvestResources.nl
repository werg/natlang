---
description: Fit eligible lot allocations into labor, machinery, storage and transport time windows.
args: { desk: HarvestDeskState; delta: FieldDelta; allocation: LotAllocation }
returns: CapacityFit
uses: [coordinateHarvest/exactHarvestArithmetic/verifyQuantityLimits]
---
Bind windows = `enumerateHarvestWindows(desk, delta, allocation)` and timeFit = `verifyCapacityAndOperators(desk, windows, allocation)` for exclusive-window conflicts and operator qualifications. When the source provides `delta.resourceQuantityLimits`, bind exactCapacityChecks = `verifyQuantityLimits(delta.resourceQuantityLimits)`; otherwise retain capacity as unresolved wherever a limit is required. Return CapacityFit with timeFit's plan and overlaps, exactCapacityChecks, quantityConserved only when all supply and resource checks pass, feasible only when both timeFit and every exact check pass, and blockers for each unsupported or exceeded resource limit. Equipment without a qualified named operator is unavailable. Do not alter accepted commitments to hide infeasibility.
