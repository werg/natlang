# Qualified native expert backend continuation

The supervisor selects a backend from actual restored-parent numerical and
execution evidence. This remains an adaptive training decision, rather than an
automatic promotion based on configured module counts.

1. Retain the running native stage, source credits, weights, moments and RNG.
2. Declare the requested expert recipe and the existing restored-model backend
   qualification flag, with the unchanged explicit numerical tolerances.
3. Before expensive model restoration, verify the recipe change with the same
   stateless expert precision transition verifier used by native SFT. Backend
   kind and non-recipe configuration must remain equal.
4. After exact restoration, execute the existing shared qualification helper.
   Failed numerics or zero executed projections cannot authorize updates.
5. Record the transition and actual result. A failed candidate leaves its
   unchanged full parent available for the already qualified continuation.

Recipe selection is the supervisor's policy. Configuration comparison, exact
state restoration, numerical verification and durable evidence are crisp
mechanisms shared by both students. This provision changes no default backend,
does not enable prefix sharing, and changes no admission-cache policy.
