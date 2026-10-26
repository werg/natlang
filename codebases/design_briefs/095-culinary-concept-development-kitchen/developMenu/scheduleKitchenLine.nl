---
description: Check station capacity and time overlap for the selected menu.
args:{ledger:KitchenLedger;sharedPrep:SharedPrep[];dishes:Dish[]}
returns:{slots:KitchenSlot[];conflicts:string[];bottlenecks:string[];}
---
Schedule supplied tasks against each station's exact capacity and required duration. Count a shared prep component once and ensure its completion precedes every dependent dish. Return overlapping intervals and the station causing each conflict. Do not assume duplicate equipment or parallel labor not present in ledger.
