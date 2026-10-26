---
args: { history: FaultRecurrence[], observations: MaintenanceObservation[], prior: AssetObservation }
returns: string[]
---
Compare the original AssetObservation against current observations using asset, symptom, and operating context. Link recurrence only when supported by evidence. Keep temporarily disappeared faults under their supplied monitoring rule and do not close from silence.
