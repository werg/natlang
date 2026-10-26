---
args: { requirements: FabricationRequirement[], measurements: ShopMeasurement[], trace: MeasurementTrace }
returns: string[]
---
Compare the documented measurement tolerance with the requirement's stated dimensions and essential constraint. If the requirement does not state an acceptance tolerance, return it as unresolved; do not invent a pass band from measurement precision alone.
