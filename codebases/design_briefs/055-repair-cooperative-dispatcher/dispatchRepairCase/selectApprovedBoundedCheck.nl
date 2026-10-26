---
args: { symptom: SymptomSeparation, repair: CooperativeRepairRequest }
returns: ApprovedRepairDiagnostic
---
Choose only a supplied procedure that addresses an unresolved hypothesis and has explicit skill and stop conditions. If a hazard requires isolation and the approved procedure does not supply it, return route specialist; if no procedure is informative, return blocked. Never invent a diagnostic instruction.
