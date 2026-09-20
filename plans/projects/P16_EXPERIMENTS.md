# P16 — Simulation and experiment laboratory

Status: proposed implementation. [Shared capabilities](README.md).

## Natlang prerequisites

C2 is fundamental: explicit seed policy and independent model/world/random-helper streams. C3 invokes supplied candidate programmes in isolated child runs. C4 records outcomes, failures and captured observations. C6 may accelerate independent trials but is not required for the first comparison. Copying simulation state is ordinary data manipulation; cloning an unfinished interpreter requires additional snapshot support and is not assumed.

## Programme and typed boundary

`compare.nl(question, world, candidates, budget) -> ExperimentReport` calls `design.nl`, `choose_cases.nl`, `analyse.nl` and `propose_followup.nl`. Exact helpers implement simulations, paired scenario allocation and metric calculations.

Records contain experiment identity, candidate/source revisions, initial world, seed namespaces, declared metrics, trial outcomes and evidence links. Large histories/native simulator instances can remain in eval. The report distinguishes completed, failed and missing trials and preserves the denominator.

## Crisp environment

A headless host supplies `experiments.run(candidate, fixture, config)`, world snapshot/restore and exact analysis. Meta-invocation uses explicit budgets and child-run links, so trials are not invisible work hidden inside one eval. Native simulators may share a process, but each trial gets independent state or a verified reset.

Candidate programmes and metrics are frozen before evaluation. Adaptive scenario selection creates a new exploratory batch; it does not modify the untouched evaluation set. Matched environment seeds are separate from policy model seeds, so a changed policy does not accidentally change exogenous randomness.

## Reduction and stream shape

Finite Map runs a known batch, initially serially. Fold aggregates typed trial outcomes. Larger interactive batches can emit progress/result events into an experiment Fold. C6 parallel execution must preserve trial inputs/random configuration and explicit outcome attribution; completion order must not choose the next trial's seed.

Branch worlds at explicit simulation checkpoints first. Time-travelling arbitrary host state or a partially executed eval is outside the first slice.

## Delivery and checks

1. Compare two scheduling/economic policies on fixed initial states and predeclared metrics.
2. Repeat identical profiles and vary completion order. Gate: seed test vectors and exact simulator metrics agree where reproducibility is promised.
3. Add missing/failed runs and bounded follow-up experiments. Gate: reports keep failures in accounting and distinguish evidence from speculation.
4. Add parallel execution after isolation and aggregation contracts pass.

Test changed candidate source, accidental shared RNG, cross-trial mutable state, interrupted batches and premature conclusions from few trials.

## Trace and teacher

Capture manifests, seed derivations, child traces and computed metrics. Teacher samples train experiment selection and evidence-based interpretation, not recomputation of exact statistics in prose. Hold out world families and policy compositions. This project supplies reusable testing infrastructure for P02 and P08, but those first slices can use simpler standalone fixtures before the full experiment interface exists.
