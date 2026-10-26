---
description: Advance contested plot allocations through proposed, trial, accepted or declined states.
args: { garden: GardenDossier; change: GardenChange; alternatives: PlotAlternative[] }
returns: AllocationNegotiation
---
Call `bindMemberResponses(garden, change, alternatives)` and bind every response to its member and exact alternative version. A suggestion remains proposed until every affected member accepts the relevant plot assignments and duties. A rejection closes only that proposal. When interests conflict without a supplied priority rule, offer a time-bounded reversible trial only if all affected members explicitly agree to its review date and reversal terms. Retain unaccepted duties visibly.
