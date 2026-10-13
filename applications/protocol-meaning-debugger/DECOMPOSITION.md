# Protocol Meaning Debugger

## The natlang program

Two research pipelines can report identical numbers while measuring different propositions. This debugger reconstructs the question each pipeline actually asks, follows semantic changes through provenance, chooses discriminating counterfactual probes, localizes the first meaningful divergence and proposes a minimal justified repair. It preserves genuinely unresolved alternatives and separates numerical coincidence from semantic equivalence.

All interpretation, causal investigation, probe selection, repair and recovery policy is NL pseudocode. Crisp seams supply exact data access, probe execution under an explicitly supplied toy measurement model, hashes, arithmetic and source verification. No helper silently decides whether two methods answer the same question.

## Target source organization

23 `.nl` modules / approximately570 NL lines, with task-specific branches and a nested probe/diagnosis path:

1. `debug.nl`: orchestrate reconstruction, investigation and repair on a snapshot.
2. `debug/intake.nl`: bind source-visible protocol, sample, execution and metric identities.
3. `readEvidence.nl`: preserve file ownership and receipt links.
4. `debug/intake/interpretQuestion.nl`: recover target population, intervention, event anchor and outcome meaning.
5. `lineage.nl`: join source artifacts to actual execution steps.
6. `lineage/resolveIdentity.nl`: distinguish hashes, aliases, descriptions and duplicated filenames.
7. `populations.nl`: interpret inclusion, exclusion and missingness without broadening the target population.
8. `timing.nl`: construct event-relative measurement times and calendar assumptions.
9. `metrics.nl`: interpret polarity, normalization and aggregation meaning.
10. `metrics/interpretMetric.nl`: invoke and consume a local NL literal under supplied measurement definitions.
11. `debug/compare.nl`: produce proposition-level equivalence and divergence obligations.
12. `debug/compare/interpretScope.nl`: judge one candidate contrast under captured target question and lineage.
13. `debug/hypotheses.nl`: build competing causal explanations consistent with current observations.
14. `probes.nl`: generate admissible single-distinction and interaction probes.
15. `probes/predict.nl`: produce hypothesis-conditioned qualitative and exact predicted observations.
16. `probes/choose.nl`: select a probe whose outcomes discriminate remaining hypotheses.
17. `debug/investigateStep.nl`: consume one finite experiment agenda item and update typed evidence.
18. `debug/investigateStep/interpretOutcome.nl`: use a scoped NL literal to interpret observed probe effects.
19. `debug/localize.nl`: identify the first meaningful divergence supported by intervening observations.
20. `debug/repair.nl`: propose changes that restore the intended question while preserving justified downstream results.
21. `debug/repair/auditRepair.nl`: replay population, timing and metric implications and verify the claimed repair.
22. `debug/report.nl`: return causal diagnosis, exact probe receipts, unresolved explanations and proposed artifact edits.

Population, timing and metric analyses are distinct branches feeding proposition comparison. Probe prediction and choice are nested under investigation; observed outcomes update the hypothesis state and determine later probes. Repair reuses the earlier branches and cannot be accepted merely because the old headline number reappears.

## Iteration and exact boundaries

The input supplies a finite probe universe and testable intervention points. Investigation uses `iterateOn` over the unexamined probe agenda; each continuing step consumes one probe and preserves its exact receipt. A no-information result is recorded, not retried forever. Stop when the causal distinction is resolved or every admissible probe is exhausted.

A declared toy-measurement service computes only the supplied exact formulas and observations. It has no causal classifier or repair policy. Source hash/quote checks, arithmetic and bounded artifact commit are crisp because they require exactness. Interpretations, admissibility, discriminating usefulness and repair choices are NL stages with typed outputs and source witnesses.

## First concrete fixture

A fictional field-device protocol asks for successful detections among commissioned devices, anchored to the accepted commissioning event. One implementation selects only devices with telemetry available at launch, anchors to calibration, and counts failures while labeling them successes. The baseline happens to show50percent under both meanings. Counterfactual trials independently exercise late telemetry, shifted calibration and outcome polarity; an archived same-named definition is a provenance decoy. At least twenty linked source facts connect population, timestamps, missingness, provenance and metric definitions.

The visible contract states all IDs, formulas, permissible probes and result semantics. Hidden oracle constraints identify supported first divergences, acceptable minimal repairs and unresolved alternatives. Type/source replay and independently reviewed semantic gold precede live full-project collection. About 48 live samples per adopted model-facing variant remain required; no reference trajectory counts as a live execution.

Status: decomposition authored before implementation. Build the first full application and inspect its live semantic contributions before expanding architecture families.

## Typed exact seams

`measurementLab.evaluate` receives an explicitly interpreted `ExecutablePlan`: exact included unit IDs, anchor times, interval bounds and numerator outcome. It filters numeric rows and divides exact counts; it never chooses eligibility, anchors, outcome meaning, probe relevance or repairs. Text services verify source hashes, exact witness spans and nonoverlapping authorized replacements. All semantic selection and causal judgment remain numbered NL.

Reused departments (`populations`, `timing`, `metrics`, `plans`) live at package root and are exposed through explicit `uses`; nested specialists retain their own callable folders. Predictions are typed materialized plans and exact receipts, while interpretation remains a returned scoped NL judgment consumed by investigation state.

`plans.nl` is the23rd module: it joins the separately returned population, timing and metric contributions into typed exact evaluation inputs. Investigation caches each immutable seed prediction once and chooses adaptively using current hypothesis status. The repair branch reconstructs definitions and uses a separate edited context, so original cached predictions cannot certify a repair.
