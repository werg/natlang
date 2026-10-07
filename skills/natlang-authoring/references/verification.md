# Verification and diagnosis

## Separate what each result establishes

| Evidence | Establishes | Does not establish |
|---|---|---|
| `natlang check` | Types, `nl` signatures, callable scoping, loop policy | That the model follows the instructions |
| Scripted interpreter run | Real runtime wiring: calls, captures, effects, completion | Semantic quality of any model |
| Exact oracle and effect assertions | Computation, IDs, counts, requested operations | Interpretations outside the assertions |
| Live interpreter run | One model executed one scenario | General reliability or cross-backend determinism |
| Independent semantic review | Evidence supports the evaluated meaning | Correctness on unseen scenarios |

Run the bundled review example through the runtime with a scripted model driver (label it as wiring evidence):

```ts
import { createNatlangRuntime, loadNatlang } from '@natlang/node';

const review = loadNatlang('review/review.nl');
const runtime = createNatlangRuntime({ model: scriptedDriver });   // or a real model driver
const report = await runtime.run(() => review(['The trial improved response times.'], 'Improved response times'));
```

A driver receives `{ messages, tools, temperature, seed, max_tokens }` and returns `{ calls: [['eval', { code }]] }`, `{ calls: [['return_result', { status: 'success', value }]] }` (or `status: 'blocked'` / `'failed'` with a `reason`), or `{ text }` (done, or a string result). For live runs, record the model identity and settings; never use fixture answers to claim model success.

## Scenario design

Tie scenarios to desired behavior: ambiguity, conflicting evidence, missing information, zero/one/many items, order-sensitive updates, interruptions, and a case that needs several inspected operations. For stateful systems verify receipts and committed state, not the model's prose. For directory reducers verify that paths stay in the folder, a direct call discards changes, and `folder.apply` keeps only the committed ones. For stochastic behavior pin source, model, template, seed, sampling, and input order, and report trial counts.

In a checkout: `npm --prefix ts-host run build`, then the relevant `ts-host/test/*.test.mjs`, `npm --prefix ts-host run test:conformance`, and `natlang check` on the project you changed.

For repeatable semantic evaluation or instruction/guidance search, use the suite and
fixture workflow in [adaptation and evaluation](adaptation.md). Keep related source
families in one split, score independent observations, and preserve infrastructure
failures separately from incorrect or blocked execution outcomes.

## Diagnose before changing policy

- `nl-unknown-return` or `nl-ambiguous-signature`: add the missing annotation or `nl<T>`; do not widen to `any` to silence it.
- `capture-conflict`: another task changed a captured `let` during the call. The model retries its eval; if it recurs, make the state a parameter or a returned value instead of a shared variable.
- `called itself without a smaller argument`: the recursive call must get a part of the input, a shorter array or string, or a smaller non-negative integer. Pass depth or a path as its own argument rather than in a new object; a graph reached by ID needs a bounded loop or `iterateOn` over a frontier. A natural-language function reached from its own call: split the responsibility.
- Loop policy error in callable-folder code: rewrite as `for...of`, a counted loop, an array method, or `iterateOn`.
- Correct type, wrong answer: improve instructions, evidence access, or decomposition; structural validation is working.
- Growing prompts: inspect repeated data and live-value previews, not just source length.
- A long call that loses track after compacting: check that its progress lives in scope variables, and that the compaction note says what is done and what is left; the full history stays in `transcript`.
- Every run fails the same way: suspect the program before the model. In teacher data, uniform failures were an instruction naming a technique without its step, a phrase that read as a finishing status, a store the model could not see how to reach, or an environment it could edit.
- A run succeeds by changing what it was meant to use (editing a checker, a board, a simulated world to accept its move): that module is outside the program; make it a service.
- A run skips the helper it was meant to use and reads the helper's data itself, correctly: the task depended on privacy. Give the helper the access (a scoped service) or accept the direct route.
- The model replies with code, or with a result written as text, instead of calling a tool: the runtime tells it so; if it persists with a model, check that its chat template renders the tools and that the server parses its tool-call syntax before changing the program.

Rejected actions and failed evals are reported to the model, which repairs them; `maxFailureRepairs` and the turn, token, and time budgets bound a call only when set. Nothing rolls back an external effect. Record the budgets used in any model study.

## Training handoff

Traces (`runtime.run(fn, { trace })`, or `fileTraceSink(dir)`) record the presented messages and tools, actions, observations, effects, seeds, and outcomes of every invocation. Teacher tasks are program IR projects (`natlang.program/2`: root `.nl`, files, services, inputs, expected). Do not admit a trajectory because it parses or reaches `done`; evaluate semantics independently and split related variants by source family. Review rejections by family, not only by reason: a family whose runs all fail alike is a task to fix, while scattered failures are the model's to learn from. A technique the model skips unprompted is taught by a hinted twin of the same case; the unhinted run is the matching negative.

Anchors: `ts-host/src/teacher/`, `ts-host/scripts/teacher-collector.mjs`, `PROGRAM_IR_PIPELINE.md`, `TEACHER_SETUP.md`.
