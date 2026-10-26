---
description: Verify each dish and shared preparation against the actual station and equipment inventory.
args:{ledger:KitchenLedger;sharedPrep:SharedPrep[];dishes:Dish[]}
returns:{taskId:string;requiredEquipment:string[];missingEquipment:string[];blockedByCapacity:string[];dependsOnSharedPrepIds:string[];evidenceIds:string[]}[]
---
For every dish and shared prep node, resolve each required equipment identifier to the supplied ledger inventory and capacity. Keep equipment type, capacity and available time window distinct; a station existing in the venue does not establish that it is free during this task's setup, service and teardown interval. Link shared prep completion to every dependent dish. Return exact missing or conflicting equipment IDs and the evidence for each capacity claim. Leave unknown labor or equipment capability unresolved; do not create a parallel station.
