---
description: Reject exact interval collisions and unstaffed equipment from the proposed harvest schedule.
args: { desk: HarvestDeskState; windows: string[]; allocation: LotAllocation }
returns: CapacityFit
---
Compare half-open time intervals for every exclusive equipment, operator, storage and vehicle resource. Check each proposed lot quantity against the recorded capacity using exact decimal arithmetic. An unstaffed machine has zero feasible capacity. Return each conflict with resource ID, overlapping interval, affected lots and source IDs; keep unaffected allocations intact.
