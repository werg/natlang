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

## Neuralese conversion

Version 2, 2026-10-04 (decisions 40–42; S5 §2.2). `ts-host/scripts/neuralese-convert-trajectories.mjs`
(`src/compiler/neuralese-conversion.ts`) converts a site when it is reused, handed from one agent to another, or large
enough that a digest saves context; values read once by the call that produced them stay text.

- **Parts.** A converted message's `content` is a list of parts: `{ "type": "text", "text" }`,
  `{ "type": "soft", "name" }` (a trainable soft parameter), `{ "type": "read", "name" }` (a block written elsewhere
  in the trajectory) and `{ "type": "digest", "name", "source", "preview" }` (a digest the digest operator writes from
  `source`, the full value; `preview` is the listing's crisp cut-off text).
- **Soft parameters**, initialised by `encode` (one forward pass through the port, no summarising call):
  `prompt:<piece>` for the runtime's prompt pieces (`src/native/system-prompts.ts`); `prompt:system@<sha12>` for
  system text of an older runtime; `guidance@<sha12>` for program guidance; `instructions@<sha12>` for instructions
  serving at least `--instructions-reuse` (2) distinct calls, shared by all of them, plus a deterministic
  `--instructions-share` (0.1) of single-use instructions for coverage. Initial texts are written once to a pieces
  file (`name`, `kind`, `text`).
- **Handover notes.** A `compact_history` call's `note` argument becomes
  `{ "$write": { "name": "handover:<sha12>", "type": "Neuralese<HandoverNote>", "source" } }`: the model writes the
  block there, `source` (the crisp note) is the teacher's view. The pinned note message becomes
  `[soft prompt:handover/open, read handover:<sha12>, soft prompt:handover/close]`.
- **Digests.** In the opening listing (`scope_0`), a value the runtime cut off becomes a digest site when the record
  holds the full value (the root call's `task.program_ir.semantics.inputs`).
- **Counts.** `neuralese_conversion.sites` gives, per site kind, the converted count and the exact count by reason:
  tool outputs and instructions (`single-use`), printed child-call results (`needs-graph-record`), digests without the
  full value (`full-value-unavailable`), `nl` literals (`later-curriculum-step`), turn-count notices (`dynamic-text`).

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
