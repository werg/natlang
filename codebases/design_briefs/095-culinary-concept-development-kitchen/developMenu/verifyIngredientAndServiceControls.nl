---
description: Verify ingredient restrictions, exact batch results and venue food controls.
args:{ledger:KitchenLedger;requirements:{mandatory:string[];preferences:string[];unverifiedIngredientIds:string[];requiredControls:string[];};tasting:{dishes:Dish[];sharedPrep:SharedPrep[];preferenceNotes:string[];repeatableDefects:string[];invalidatedDishIds:string[];};scaledBatches:ScaledBatch[]}
returns:MenuFinding[]
---
For every served dish trace ingredient lots to current verified facts and approved handling controls. Block unverified substitutions and any allergen conflict. Exact scaled quantities and cost must reconcile to inventory and budget. An allergen-free representation remains blocked unless venue-specific separation and qualified approval are supplied. Keep subjective tasting findings for chef judgment.
