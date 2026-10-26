---
description: Hold tasks requiring a broken shared tool without erasing existing reservations.
args: { garden: GardenDossier; change: GardenChange; negotiation: AllocationNegotiation }
returns: SeasonBoard
---
Identify only tasks whose accepted plan requires a tool listed as broken in the current change. Preserve their original reservation and mark repair dependency. Keep unrelated tasks scheduled. Do not select an unapproved substitute tool or reassign an owner's task.
