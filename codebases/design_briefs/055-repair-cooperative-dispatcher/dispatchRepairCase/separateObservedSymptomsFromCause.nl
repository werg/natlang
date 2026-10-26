---
args: { repair: CooperativeRepairRequest, prior: RepairSymptom }
returns: SymptomSeparation
---
Extract observed behavior and prior attempts without accepting the requester's diagnosis as fact. Keep hypotheses and missing attribution separate; set isolationNeeded when any supplied risk flag requires it.
