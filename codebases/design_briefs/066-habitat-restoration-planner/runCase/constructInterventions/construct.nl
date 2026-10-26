---
args: { decisions: DecisionAtom[], prepared: PreparedEvidence, input: CaseInput }
returns: DraftRestorationOption
---
Construct a RestorationOption draft with its actual domain fields. Only supported values enter `values`; preserve conflicts, missing fields and evidence bindings. Propose staged interventions from approved guidance and known site conditions. Explain the hypothesized mechanism, dependencies, resource needs, and stop conditions; route unfamiliar or potentially harmful interventions to qualified specialists instead of improvising.
