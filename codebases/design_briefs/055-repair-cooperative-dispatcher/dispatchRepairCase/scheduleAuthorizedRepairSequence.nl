---
args: { assignments: RepairerMatch[], repair: CooperativeRepairRequest }
returns: RepairSequence
---
Order only approved checks and assigned tasks by their prerequisites. Reject a tool window already reserved, accept a substitute part only with compatibility evidence, and stop when any new cost or scope exceeds the customer authorization. Return workshop work only if all checks pass; otherwise hold or route to the specialist process.
