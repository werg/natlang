---
description: Expose uncertainty that remains after the requested mechanism is identified.
args:
  posterior: Posterior
  policy: PolicyResult
  request: LaboratoryRequest
returns: ResidualUncertainty[]
---
1. Visit each solved leaf reachable through actual selected policy transitions. Retain its actual
   positive-mass worlds rather than substituting the root posterior or one representative world.
2. Compare those worlds at the requested goal boundary. Record the surviving theory IDs and whether
   the mechanism is known; separate their accession genotype alternatives into explicit collections.
3. Explain why differing genotypes can remain consistent with a solved mechanism. Cite the source
   theory/readout witnesses that made those worlds observationally equivalent on this branch.
4. If a leaf still contains multiple requested mechanisms, report mechanismKnown false. A numerical
   policy value or a confident leaf label does not establish goal completion.
5. Return the residual records with actual node/accession IDs and exact evidence. Preserve unresolved
   biological facts without turning them into extra experiments that the goal never requested.
