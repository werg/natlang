---
description: Reconcile every served dish's ingredient lots, allergens and verified handling controls.
args:{ledger:KitchenLedger;requirements:{mandatory:string[];preferences:string[];unverifiedIngredientIds:string[];requiredControls:string[]};dishes:Dish[]}
returns:MenuFinding[]
---
For each dish resolve every recipe ingredient to the current IngredientFact and lot. Compare verified allergens and dietary properties with mandatory restrictions, then check approvedControls against venue handling, separation and service conditions. A missing lot, stale fact, unverified substitute or unsatisfied cross-contact control blocks the affected dish. Emit findings bound to dishId, ingredientId/lotId and evidence IDs; do not infer safety from a dish label or a recipe name. Preserve preferences as review findings rather than safety clearance.
