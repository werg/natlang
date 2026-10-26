---
args: { requirements: FabricationRequirement[], measurements: ShopMeasurement[] }
returns: MeasurementQualification
---
Bind trace = verifyMeasurementTraceability(requirements, measurements). Bind tolerances = testRequiredToleranceFit(requirements, measurements, trace). Admit a measurement into production dimensions only when its feature, unit, method, verification status, and source all match the requirement. Return missing, unverified, and tolerance-conflict details separately; approximate dimensions remain provisional.
