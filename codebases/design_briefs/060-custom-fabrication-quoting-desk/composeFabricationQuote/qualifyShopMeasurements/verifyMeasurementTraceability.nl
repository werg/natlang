---
args: { requirements: FabricationRequirement[], measurements: ShopMeasurement[] }
returns: MeasurementTrace
---
Match each requirement to an actual measurement record for the same feature. Retain original measured value and unit. A verified flag without evidenceRef, or a measurement method that does not match the required feature, is not traceable production data.
