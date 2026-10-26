---
args: { pendingIds: string[], mandates: ProtocolMandate[], resources: ResourceAssessment[] }
returns: string
---
Choose the pending experiment with the fewest eligible instrument/operator windows. Break ties by earliest hard sample deadline, then explicit dependency order from the protocol. A requester preference may not change this branch order before hard constraints are considered.
