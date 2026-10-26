---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftScaffoldingPlan
---
Construct a ScaffoldingPlan draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Interpret progress artifacts to identify where support is needed without doing the substantive work for learners. Provide targeted examples, prompts, or task decomposition, and remove support when evidence shows independent capability.
