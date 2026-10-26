---
args: { impacts: InterventionImpact[], interventions: RestorationIntervention[], experts: ExpertReviewScope[] }
returns: EcologicalIncidentReview
---
1. For each supplied impact, bind linkage = linkImpactToIntervention(impact, interventions). Preserve the exact observation and source reference; ambiguous or unlinked effects remain review items and must not be reassigned to an unrelated intervention.
2. If linkage.effect is adverse or unclear, bind route = routeToQualifiedExpert(impact, linkage, experts). Suspend only a matched intervention with an adverse or unresolved impact. A benign result does not suspend work.
3. Keep an expert request open when no currently available expert covers the affected system; do not substitute an unqualified volunteer. Return the exact suspended intervention IDs, each scoped expert request, and all preserved observation references.
