# P09 — Spell interpretation game and specialised student

Status: proposed implementation. [Shared capabilities](README.md).

## Natlang prerequisites

The finite first slice needs C0/C7 only: semantic functions, typed records/unions and exact validation. Use C2 and C4 for reproducible experiments and trace-linked training. C5 later consumes player/world events. There is no demonstrated need for a host-type extension or special spell syntax in natlang.

## Programme and typed boundary

`cast.nl(utterance, observation, caster, rules) -> SpellResult` calls `interpret.nl`, `compose.nl`, `resolve_reference.nl` and `describe.nl`. Exact helpers calculate cost, validate effects and apply an accepted plan. Define a finite initial effect vocabulary, then expand deliberately.

A `SpellPlan` holds target IDs, ordered effects, magnitude/duration parameters and expected world revision. Use ordinary typed lists for compositional plans. `SpellResult` distinguishes applied, needs-clarification and impossible/invalid outcomes. Native world/animation objects never need to appear in these values.

## Crisp environment

A simulator binding exposes exact preview/validate/apply operations. A headless first implementation can use pure crisp functions over world records. A graphical host later retains native world and effect objects in its selected engine environment. The same typed plan boundary allows both without pretending their render behavior is identical.

Application is conditional on the recorded world revision and current cost/target validity. Exact code atomically applies the validated plan or returns a changed-world result. It must not silently reinterpret a spell the model proposed or partially spend resources before discovering a preventable validation failure.

## Reduction and stream shape

One cast is an ordinary bounded function call. Interactive play folds utterance/world events; a cast based on stale observations must be revalidated or replanned in a later step. Expensive interpretations can finish asynchronously, but result revision checks remain exact. Fine-tuned model selection is run configuration, not a choice the student makes inside `cast`.

## Delivery and checks

1. Small arena with movement, damage, shielding and a few elemental effects. Gate: lawful plans and exact resource conservation.
2. Add compositional utterances, exclusions and target ambiguity. Gate: intent rubric plus exact simulator checks; impossible requests produce honest results.
3. Collect teacher scenarios with frozen rules and train a spell-specific student. Gate: held-out mechanic combinations and phrasing, paired comparison with the general interpreter.
4. Connect the graphical stream host and verify bounded decision latency and stale-result handling.

Include metaphor, negation, nonexistent targets, contradictory requests, insufficient resources and adversarial attempts to redefine the rules.

## Trace and teacher

Capture the original utterance, visible world snapshot, rule/source version, proposed plan, checks and committed actions. Never admit a semantically wrong plan merely because it was legal. Hold paraphrase/composition families together across splits. Track autonomous success and clarification/error quality separately. This is an early application precisely because it can provide rich training evidence without waiting for a large infrastructure layer.
