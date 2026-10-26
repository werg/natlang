---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftPlotConstraint
---
Construct a PlotConstraint draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Reconcile plot records, field observations, and supplied gardening guidance to describe known conditions and constraints. Preserve uncertainty about soil or other untested properties, and route consequential interventions to appropriate local expertise.
