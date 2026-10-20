# Held quality under declared CE supervision

`interpretQuality(rawHeld, declaredHeld, taskOutcomes) -> Continuation` is a typed
natural-language policy: retain historical telemetry, inspect actual target
support and task behavior, diagnose regressions, and preserve useful updates.
A raw prompt-dominated CE score cannot silently choose the production policy.

1. `declareHeldIntent(immutableHeldSources) -> HeldDeclarations` uses the same
   explicit source encodings as training, in a separate `held_sources` list.
   Crisp bindings authenticate source/split/coordinates, never training credit.
2. `measureHeld(graph, declarations) -> QualityEvidence` executes one actual
   graph and each readout once. Crisp reductions report original CE unchanged,
   declared-mass CE and exact target-only CE. No inferred target thresholds.
   Uncovered sources remain visibly unmeasured for these additional metrics.
3. `recordQuality(evidence) -> Report` retains exact source and declaration
   bindings, coverage and legacy metric semantics. No exposure is granted.
4. `publishRecovery(state) -> Checkpoint` always retains newest full optimizer,
   RNG and exposure. With declared CE active, raw CE no longer nominates an
   optional best export; task/target quality remain supervisor inputs.
5. `evaluateSemanticHeld` strips TRAIN-only CE declarations before measuring
   historical held objective, avoiding both accidental reweighting and an
   undeclared-held-source failure. Semantic target telemetry must use explicitly
   held-bound declarations, never the training source list.

No model-facing text, task prompt, held membership, reward, or runtime gate
threshold changes. Existing exact final coverage and task/runtime checks stay.
