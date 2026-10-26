---
description: Derive the resource intervals required by each feasible lot allocation from the round's sources.
args: { desk: HarvestDeskState; delta: FieldDelta; allocation: LotAllocation }
returns: string[]
---
For each accepted allocation, collect the grower collection window, equipment booking, named operator shift, cold-storage interval, and transport slot from current evidence. A missing window remains unresolved. Never treat vehicle capacity as available unless the vehicle and a qualified operator are both confirmed for that interval.
