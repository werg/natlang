# Verification and diagnosis

## Separate what each result establishes

| Evidence | Establishes | Does not establish |
|---|---|---|
| `natlang check` | Types, `nl` signatures, callable scoping, loop and recursion policy | That the model follows the instructions |
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

A driver receives `{ messages, tools, temperature, seed, max_tokens }` and returns `{ calls: [['eval', { code }]] }`, `{ calls: [['return_result', { value }]] }`, or `{ text }` (done, or a string result). For live runs, record the model identity and settings; never use fixture answers to claim model success.

## Scenario design

Tie scenarios to desired behavior: ambiguity, conflicting evidence, missing information, zero/one/many items, order-sensitive updates, interruptions, and a case that needs several inspected operations. For stateful systems verify receipts and committed state, not the model's prose. For directory reducers verify that paths stay in the folder, a direct call discards changes, and `folder.apply` keeps only the committed ones. For stochastic behavior pin source, model, template, seed, sampling, and input order, and report trial counts.

In a checkout: `npm --prefix ts-host run build`, then the relevant `ts-host/test/*.test.mjs`, `npm --prefix ts-host run test:conformance`, and `natlang check` on the project you changed.

## Diagnose before changing policy

- `nl-unknown-return` or `nl-ambiguous-signature`: add the missing annotation or `nl<T>`; do not widen to `any` to silence it.
- `capture-conflict`: another task changed a captured `let` during the call. The model retries its eval; if it recurs, make the state a parameter or a returned value instead of a shared variable.
- Recursion error: a function reached itself through its callers. Restructure into iteration (`iterateOn`) or split the responsibility.
- Loop policy error in callable-folder code: rewrite as `for...of`, a counted loop, an array method, or `iterateOn`.
- Correct type, wrong answer: improve instructions, evidence access, or decomposition; structural validation is working.
- Growing prompts: inspect repeated data and live-value previews, not just source length.
- Repeated work after conversation rollover (when segmentation is configured): check that progress lives in the scope and that effects are not restarted.

Rejected actions and failed evals are reported to the model, which repairs them; `maxFailureRepairs` and the turn, token, and time budgets bound a call only when set. Nothing rolls back an external effect. Record the budgets used in any model study.

## Training handoff

Traces (`runtime.run(fn, { trace })`, or `fileTraceSink(dir)`) record the presented messages and tools, actions, observations, effects, seeds, and outcomes of every invocation. Teacher tasks are program IR projects (`natlang.program/2`: root `.nl`, files, inputs, expected). Do not admit a trajectory because it parses or reaches `done`; evaluate semantics independently and split related variants by source family.

Anchors: `ts-host/src/teacher/`, `ts-host/scripts/teacher-collector.mjs`, `PROGRAM_IR_PIPELINE.md`, `TEACHER_SETUP.md`.
