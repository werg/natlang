---
description: Check station capacity and time overlap for the selected menu.
args:{ledger:KitchenLedger;sharedPrep:SharedPrep[];dishes:Dish[]}
returns:{slots:KitchenSlot[];conflicts:string[];bottlenecks:string[];}
---
Bind compatibility = `assessDishStationCompatibility(ledger, sharedPrep, dishes)` before scheduling. Schedule only tasks whose returned equipment links and shared-prep dependencies resolve. Preserve each missing equipment ID and capacity block as a bottleneck; do not interpret another dish's station use as free capacity. Schedule supplied tasks against exact intervals and capacity, count each shared prep component once, ensure its completion precedes every dependent dish, and return both pairwise interval conflicts and compatibility findings.
