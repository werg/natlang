# P16 — Simulation and experiment laboratory

Status: finite laboratory implemented in [codebases/experiment_lab](../../codebases/experiment_lab/README.md) and [applications/experiment_lab.py](../../applications/experiment_lab.py). [Shared capabilities](README.md). Natlang selects cases and interprets exact results; the host runs isolated trials, pins seeds/source revisions, records traces and an append-only journal, and verifies report metrics. Trial labels are domain-neutral. Backends now cover semantic merge, type inference, dependency-plan testing and SQLite migration. The case menu is finite, while model runs have no default episode or time limit. Live teacher quality and semantic review remain open.

The first unbounded Bonsai type trial completed after more than seven minutes
and passed its frozen signature rubric. The old JSON-text chat adapter then
repeated a `summarize` call without the required `trials` input. A typed chat
adapter compiled the existing natlang alternatives into separate exact tools;
the captured first report turn then supplied both required inputs. Naively
splitting every field exceeded the server's earlier 12,288-token launch context,
so the adapter groups only destinations with identical value schemas or copy
source sets and uses short descriptions. With a 32,768-token launch context,
the complete typed report replay passed its exact metrics and output assertions.
The peak prompt was 14,949 tokens and the largest tool menu about 40 KB; reducing
that repeated schema cost is the next teacher-harness efficiency task.
The cost comes from re-sending expanded `write` and `call` alternatives after
each workspace change. Measure schema bytes and prompt tokens per turn, then
compare lossless grouping and a backend-only staged constrained decoder. Keep
the natlang action set and exact destination/type checks unchanged. Do not
replace schema cost with an arbitrary run-length cap.

Operationally, separate evaluation processes can submit to the same one-slot
teacher server at once. This interleaves long trials and makes wall time hard to
attribute even when neither process leaks memory. Give the *outer* evaluation
launcher an endpoint-capacity lease or queue, with explicit capacity per target;
do not put scheduling into natlang's core reduction semantics. Preserve partial
journals and traces before any deliberate cancellation. Monitor available disk
space during artifact-heavy training runs, and use an explicit retention policy
for generated datasets rather than silently deleting evidence.

## Natlang prerequisites

C2 is fundamental: explicit seed policy and independent model/world/random-helper streams. C3 invokes supplied candidate programmes in isolated child runs. C4 records outcomes, failures and captured observations. C6 may accelerate independent trials but is not required for the first comparison. Copying simulation state is ordinary data manipulation; cloning an unfinished interpreter requires additional snapshot support and is not assumed.

## Programme and typed boundary

`compare.nl(question, world, candidates, budget) -> ExperimentReport` calls `design.nl`, `choose_cases.nl`, `analyse.nl` and `propose_followup.nl`. Exact helpers implement simulations, paired scenario allocation and metric calculations.

Records contain experiment identity, candidate/source revisions, initial world, seed namespaces, declared metrics, trial outcomes and evidence links. Large histories/native simulator instances can remain in eval. The report distinguishes completed, failed and missing trials and preserves the denominator.

## Crisp environment

A headless host supplies `experiments.run(candidate, fixture, config)`, world snapshot/restore and exact analysis. Meta-invocation retains child-run links and optional explicit budgets, so trials are not invisible work hidden inside one eval. Native simulators may share a process, but each trial gets independent state or a verified reset.

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
