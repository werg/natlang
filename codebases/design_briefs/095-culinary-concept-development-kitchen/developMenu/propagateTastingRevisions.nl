---
description: Apply tasting notes at their smallest affected component and trace dependents.
args:{ledger:KitchenLedger;change:string[];composition:{dishes:Dish[];sharedPrep:SharedPrep[];coverageGaps:string[];};resourceCheck:{slots:KitchenSlot[];conflicts:string[];bottlenecks:string[];}}
returns:{dishes:Dish[];sharedPrep:SharedPrep[];preferenceNotes:string[];repeatableDefects:string[];invalidatedDishIds:string[];}
---
Classify notes as subjective preference or repeatable service/recipe defect. Revise only the named dish or shared component. If shared, propagate its changed recipe revision to every dependent dish, batch calculation, dietary disclosure and schedule slot. Return exact invalidation list; do not turn preference into a safety gate.
