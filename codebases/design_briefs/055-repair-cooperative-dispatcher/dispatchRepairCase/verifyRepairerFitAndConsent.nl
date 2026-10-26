---
args: { diagnostic: ApprovedRepairDiagnostic, repair: CooperativeRepairRequest }
returns: RepairerMatch[]
---
For each required task, match only a named repairer whose verified skill and qualification scope cover the diagnostic and whose qualification remains current for the requested work date. Preserve other valid skills when one expires. Require task consent, availability, and named supervision for trainees; return no match for unmet prerequisites.
