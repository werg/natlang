# Verification and diagnosis

## Separate what each result establishes

| Evidence | Establishes | Does not establish |
|---|---|---|
| Load/type check | Source graph and signatures are accepted | The model follows the algorithm |
| Scripted interpreter run | Real runtime plumbing and selected tool sequences work | Semantic quality of a teacher/student |
| Exact oracle/effect assertions | Computation, IDs, counts, or requested operations match expectations | Interpretations outside the assertions |
| Live interpreter trajectory | A particular model executed a particular scenario | General reliability or cross-platform determinism |
| Independent semantic review | Evidence supports the evaluated meaning | Correctness for unseen scenarios |

Load the bundled review example through the TypeScript host:

```ts
const result = await host.run({
  source: { kind: 'file', path: 'review/review.nl' },
  inputs: {
    observations: ['The trial improved response times.'],
    criterion: 'Evidence of improved response times',
  },
  modelTurn,
});
```

This loads source and executes a run. For a fixture run use a scripted session agent and label it as such. Crisp helpers can execute without a model. For a live run supply the selected model driver and record its actual identity; never use fixture answers to claim model success.

## Scenario design

Use scenarios tied to desired behavior rather than only a happy-path demo. Cover ambiguity, conflicting evidence, missing information, zero/one/many items, order-sensitive updates, and interruptions where applicable. Include a case requiring multiple inspected operations. For stateful systems verify actual receipts and committed state; a model's prose about a completed effect is insufficient.

For filesystem work, verify that ordinary lambdas have no file tools, reducer
paths stay within the supplied folder, direct reducer calls discard changes,
and `folder.apply` retains only the selected changes. For typed keyed inputs,
verify both empty and populated `Record<string, T>` values at function
boundaries.

For stochastic behavior pin source, model/template, seeds, sampling, and ordered inputs, and report how many trials were run. Keep model sampling randomness separate from game/world randomness. Classify failures: source/loader, schema/transport, runtime, continuation, host/effect, model semantics, environment capacity.

In a checkout, build with `npm --prefix ts-host run build` and run the applicable files under `ts-host/test/`. Run checks relevant to the change plus required project gates.

## Diagnose before modifying policy

- Failed eval or function call: use its diagnostic and the typed source contract to correct the operation. Do not revive path-based scope binding or unwrap malformed values to make the call pass.
- Correct return type, wrong answer: improve semantic criteria, algorithm, evidence access, or model; structural validation is working as intended.
- Growing prompt: inspect presented schemas and repeated data, not just source length. Compact representation before adding language size limits.
- Unexpected assignment or mutation failure: check the declared value shape and eval diagnostic; function parameters and locals are mutable within their call.
- Repeated work after rollover: check durable locals, restored pending nodes, marks, notes, and effect observations. Do not restart effects merely because conversation history changed.
- Capacity failure: inspect actual loaded context per server slot, model limits, and simultaneous workloads. A configured total context may be divided across slots. Never infer ownership or queue position from aggregate metrics alone.
- Low throughput: measure inference, waiting, host I/O, and context reconstruction separately. Avoid repeatedly announcing speculative progress from a quiet process.

`local` validation feedback lets the model inspect rejected actions and repair; `caller` returns diagnostics to the caller. This is a policy decision, not a correctness bypass. Neither repairs an external effect nor rolls it back. Record the selected policy for model studies.

## Training handoff

Capture source revision, typed inputs, exact presented messages/tools, actions, observations, seeds, continuations, model identity/settings, engine authority, and actual effects. Retain failed examples with clear labels. Do not automatically admit a trace because it parses, reaches `done`, or passes structural checks. Independently evaluate semantics and verify provenance. Split related variants by source program/family to avoid train/test leakage.

Repository anchors: `ts-host/src/native/scenario.ts`, `ts-host/studio/research/evaluation.mjs`, `PROGRAM_IR_PIPELINE.md`, and `TEACHER_SETUP.md`. Use current runner flags rather than copying historical model-specific launch commands.
