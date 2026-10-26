---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftQualityActionPlan
---
Construct a QualityActionPlan draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Select approved containment and corrective actions, assign qualified owners, and specify verification criteria before implementation. Distinguish immediate correction of defective units from prevention of recurrence and preserve unaffected production where evidence supports it.
