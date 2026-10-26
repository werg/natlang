---
description: Fit eligible lot allocations into labor, machinery, storage and transport time windows.
args: { desk: HarvestDeskState; delta: FieldDelta; allocation: LotAllocation }
returns: CapacityFit
---
Call `enumerateHarvestWindows(desk, delta, allocation)` and bind the resulting exact resource intervals. Call `verifyCapacityAndOperators(desk, windows, allocation)`. Equipment without a qualified named operator is unavailable. Reject overlapping use of a vehicle, storage bay or labor slot and retain the bottleneck that caused each rejection. Do not alter accepted commitments to hide infeasibility.
