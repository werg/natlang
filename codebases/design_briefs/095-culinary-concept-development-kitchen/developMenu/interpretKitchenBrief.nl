---
description: Separate mandatory dietary constraints from preferences and verified ingredient facts.
args:{brief:KitchenBrief;change:string[];ingredients:IngredientFact[];sourceRecord:MenuProject;sourceEvidence:EvidenceItem[]}
returns:{mandatory:string[];preferences:string[];unverifiedIngredientIds:string[];requiredControls:string[];}
---
Take venueId, creativeBrief, ingredientInventoryIds, serviceWindow and chefOwnerId from the top level of the MenuProject record. They govern the kitchen, time window, responsible chef and ingredient lot set. Match sourceRecord.evidenceIds and sourceRefs to sourceEvidence before treating a restriction or substitution as verified; a placeholder brief cannot relax an allergy or service constraint.

Retain each supplied dietary restriction as mandatory only when the client or authorized service record says so. Preserve preferences separately. Match ingredient facts to exact lot and revision; an unverified lot remains blocked. Identify required kitchen controls from authoritative venue evidence and do not infer them from a recipe name.
