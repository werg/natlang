---
args: { selectedId: ID | null, concepts: ShopConcept[], measurements: MeasurementQualification, finishes: FinishReference[], samples: Ref[] }
returns: ConceptSelection
---
Select only the requested concept when it exists. Bind finish = confirmPhysicalFinishReference(concepts, finishes, samples). A feature lacking accepted measurement or physical/reference approval remains a blocker; a screen rendering alone is not acceptance.
