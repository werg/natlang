---
args: { world: TrainingWorld, candidateWorld: TrainingWorld }
returns: InvariantCheck
---
Compare each resource's candidate quantity and status with its typed legal transition bounds. Reject impossible negative stock, unrecorded transfers, or a status change with no action evidence. State the exact resource and violated bound; do not repair a simulated action by silently altering unrelated inventory.
