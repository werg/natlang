# Training-data rewrites for Neuralese

Version 1, 2026-10-03. Normative for training-data builders. Registered as the API
migration [`training/api-migrations/neuralese-language.json`](../training/api-migrations/neuralese-language.json).

Model-written natlang in training data is rewritten by the compiler so that the
model learns two habits Neuralese depends on: annotate types eagerly, and name
captures explicitly. Both rewrites apply to model-neutral records
(`natlang.program/2` trajectories and their materialized turns); rendering into
any model's chat template stays a separate, later step.

## Eager typing

Every declaration the model writes in eval code carries its type annotation.

- **Input.** A materialized trajectory's eval code, with the call's scope types.
- **Pass.** For each `const`/`let` declaration without an annotation, insert the
  type the natlang checker infers, written in natlang type syntax with the call's
  aliases. Function-valued declarations get their full signature. Destructuring
  patterns get an annotation on the whole pattern.
- **Skip.** Declarations whose inferred type is `any`, `unknown`, or not
  expressible in natlang type syntax are left alone and counted.
- **Check.** The rewritten eval must compile in the same scope with the same
  diagnostics as the original, and the recorded execution must replay unchanged.

Every Neuralese literal must have a contextual type (`neuralese-untyped-literal`).
Eager typing puts the annotation in front of the literal in the token stream, where
the writer and its stop head see it.

## Explicit captures

Inline `nl` calls that capture by name mention are rewritten to `nl.with({ … })`.

- **Input.** The inline `nl` sites of a trajectory, with the compiler's capture
  analysis (the bindings each template's instructions mention).
- **Pass.** Rewrite ``nl`…` `` to ``nl.with({ a, b })`…` `` listing exactly the
  captured bindings. A captured `let` that the call writes back becomes
  `live(x)`. Function-typed captures are listed as plain (snapshot) captures.
- **Pairs.** Each rewritten site yields a text function literal with explicit
  captures. When S5 substitutes a soft body for the text, the same capture object
  carries over, giving paired text and soft function literals with checked
  captures.
- **Check.** The rewritten program must compile, the capture set must equal the
  original analysis, and the recorded execution must replay unchanged, including
  `let` write-backs.

## Literal rendering

Model-neutral records hold soft values only in reference form
(`{ "$neuralese": { "type", "id" } }`) with their blocks in `.nz` files or the
tensor store. The per-model render step turns references into marker-wrapped
payload positions ([NEURALESE_PORT.md](NEURALESE_PORT.md)); nothing upstream of it
contains control tokens.

## Lineage

Rewritten records keep their source record's ID, lineage, licence and split group,
and record the migration ID and the compiler revision that produced them. Records
built under different runtime versions are not mixed in one training build.
