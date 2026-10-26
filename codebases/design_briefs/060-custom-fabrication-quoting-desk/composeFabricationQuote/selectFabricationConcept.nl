---
args: { selectedId: ID | null, concepts: ShopConcept[], measurements: MeasurementQualification }
returns: ConceptSelection
---
Select only the requested concept ID when it exists. Identify features with missing or unaccepted measurement and require an accepted physical sample or approved reference for finish; a screen rendering alone is not customer acceptance.
