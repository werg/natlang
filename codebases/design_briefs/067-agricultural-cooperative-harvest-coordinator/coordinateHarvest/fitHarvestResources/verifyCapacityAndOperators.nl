---
description: Reject exact interval collisions and unstaffed equipment from the proposed harvest schedule.
args: { desk: HarvestDeskState; windows: string[]; allocation: LotAllocation }
returns: CapacityFit
---
Compare half-open time intervals for each exclusive equipment, operator, storage and vehicle resource, retaining the exact IDs and time evidence in every overlap. Keep operator qualification and interval assignment decisions here; numeric quantity limits are checked by the separate exact helper in the parent `fitHarvestResources` operation. An unstaffed machine has zero feasible capacity. Return each time conflict with resource ID, overlapping interval, affected lots and source IDs; keep unaffected allocations intact.
