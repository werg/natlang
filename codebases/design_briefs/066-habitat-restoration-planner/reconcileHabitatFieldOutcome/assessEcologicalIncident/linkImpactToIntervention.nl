---
args: { impact: InterventionImpact, interventions: RestorationIntervention[] }
returns: ImpactLinkage
---
Resolve the report to its named intervention and affected action area. Keep status ambiguous when its area or mechanism matches several actions, and unlinked when no intervention supports the attribution. Classify the ecological effect from the actual observation, severity and evidence as adverse, unclear, or benign. Do not infer benefit or harm from intervention completion alone; preserve report ID and evidence reference.
