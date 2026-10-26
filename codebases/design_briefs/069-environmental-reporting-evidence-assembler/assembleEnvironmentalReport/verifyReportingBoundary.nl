---
description: Exact counterfactual gate for approved environmental reporting scope and effective period.
args: { draft: DraftReportingBoundary; prepared: PreparedEvidence; decisions: DecisionAtom[] }
returns: Qualification
readout: decision
---
Pass only when included and excluded operations, measurement rules, effective scope and approval are explicit and supported by current evidence. Fail when the draft contradicts the approved boundary or selects a scope based on a favorable total. Return needs-review when boundary precedence, effective date, authority or decisive evidence is missing. Return only the finite verdict.
