---
description: Schedule bounded seasonal work only from accepted agreements and current shared-tool availability.
args: { garden: GardenDossier; change: GardenChange; negotiation: AllocationNegotiation }
returns: SeasonBoard
---
Call `holdBrokenToolTasks(garden, change, negotiation)` to mark dependent tasks held while preserving their prior reservations. Call `transferDepartingMemberCare(garden, change, negotiation)` when membership changes. A successor must accept care duties and harvest rights through a documented agreement. Keep proposed reminders, accepted tasks, and completion receipts distinct; a missed task triggers renegotiation, never automatic exclusion.
