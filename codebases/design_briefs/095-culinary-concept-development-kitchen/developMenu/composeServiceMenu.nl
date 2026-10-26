---
description: Balance menu variety and shared preparation dependencies as a service plan.
args:{ledger:KitchenLedger;requirements:{mandatory:string[];preferences:string[];unverifiedIngredientIds:string[];requiredControls:string[];};dishes:Dish[]}
returns:{dishes:Dish[];sharedPrep:SharedPrep[];coverageGaps:string[];}
---
Compare dishes as a service sequence for dietary coverage, flavor/texture variety, prep effort, equipment and pacing. Represent a shared component once with total quantity and dependent dish IDs. Keep an unmet required diet visible if no feasible dish covers it; do not assert menu readiness through slot filling.
