---
description: Reassess each affected lot's maturity, variety, quality and readiness window from current evidence.
args: { desk: HarvestDeskState; delta: FieldDelta }
returns: HarvestReadiness[]
---
For each changed lot, preserve its exact variety and lot identity. Use only current, lot-specific evidence to describe maturity and quality. Keep rejected quantity and buyer grade distinctions explicit. If evidence is informal or contradictory, set uncertainty to the conflicting source IDs and do not describe the lot as confirmed available. Retain untouched prior lot readiness records unchanged.
