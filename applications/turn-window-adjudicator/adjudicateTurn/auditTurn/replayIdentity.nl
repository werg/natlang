---
description: Independently follow instance and controller identity through actual window changes.
args:
  captured: CapturedTurn
  settled: WindowState
  source: TurnSource
returns: TurnFinding[]
---
1. Start from original initialBoard objects and captured ReferenceBindings. Read each committed before/
   after snapshot in order; identify changes by instanceId while retaining its separate stable cardId.
2. Judge each same-instance controller change against its actual submitted or triggered source effect.
   Verify unrelated properties survive. A captured controller remains its original field even after the
   object's current controller changes, and attachment ownership stays separate from attached target.
3. At each recorded departure, compare lastKnownController with the immediately preceding instance
   controller and the actual batch evidence. Retain the old discarded instance in the lineage history.
4. For reentry, require the new source-authorized instance identity and stated initial properties.
   Distinguish new state from old-instance damage/growth/attachments; any inherited property needs an
   explicit granting clause. Shared token status follows its source capacity, not the new garden stay.
5. Follow each queued trigger recipient back through its returned controllerBasis and qualifying part
   or departure. A later new instance cannot rewrite that already fixed recipient. Report the earliest
   inconsistent window with exact source evidence and the affected slot IDs.
6. Return source-grounded defended, defect or unresolved findings for these identity transitions.
   This audit diagnoses the actual proposal; it does not patch snapshots to make the lineage coherent.
