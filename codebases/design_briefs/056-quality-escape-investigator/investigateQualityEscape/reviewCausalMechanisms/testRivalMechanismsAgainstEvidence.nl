---
args: { rivals: MechanismSet, comparison: DefectRateComparison, evidence: Evidence[] }
returns: MechanismTest[]
---
For each rival mechanism, compare its predicted pattern with supporting and contradictory source evidence. Do not treat an inspection-rate trend as causal when sample frames differ. Preserve tests whose predictions remain unresolved instead of selecting the most convenient narrative.
