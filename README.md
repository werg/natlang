# natlang

natlang is TypeScript with natural-language functions. A natural-language
function is an ordinary async, typed function whose body is instructions; a
model — possibly a small local one — executes them in a persistent TypeScript
scope. Everything else is ordinary TypeScript.

```ts
import { nl } from '@natlang/node';
import classify from './classify.nl';          // a named natural-language function

export async function triage(tickets: string[], rubric: string): Promise<Report> {
  const labels = await Promise.all(tickets.map(ticket => classify(ticket, rubric)));
  const urgent = tickets.filter((_, i) => labels[i] === 'technical');
  const summary: string = await nl`Summarize what is going wrong across urgent, most severe first.`(urgent);
  return { urgent: urgent.length, summary };
}
```

The compiler plans every `nl` call before the model runs — its parameters,
return type, and the variables its instructions mention — so natural-language
calls are type-checked like any other code. A named `.nl` function may call only
the helpers in its companion folder; inline calls in application code see the
nearest `natlang.d/` folder. Host capabilities are typed services, and every
invocation is traced.

## Get started

```sh
scripts/setup_dev.sh --node-only        # install and build ts-host; installs the `natlang` command
natlang doctor
natlang run applications/evidence       # a citation-checked question-answering console
natlang check examples/triage           # type- and policy-check a project
```

`natlang setup` prepares a managed local model runtime; see
[development setup](DEV_SETUP.md) for model profiles, the browser build, and
tests.

## Read next

| Guide | Contents |
|---|---|
| [Language specification](spec/SPEC.md) | Functions, callable folders, captures, iteration, services, the model surface |
| [Skills](skills/README.md) | Authoring and integration guidance for coding agents |
| [Native packages](NATIVE_PACKAGES.md) | `@natlang/node`, `@natlang/browser`, the CLI, and distribution packages |
| [TypeScript runtime](ts-host/README.md) | Building and using the runtime and compiler |
| [Applications](applications/) | Wiki, notebook, evidence, logs, terminal, publisher, games, and more |
| [Training](TRAINING.md), [teacher setup](TEACHER_SETUP.md), [program IR](PROGRAM_IR_PIPELINE.md) | Teacher data, student training, and evaluation |

The design notes behind the TypeScript-native runtime are in
[`docs/TS_INLINE_HOST_INTEGRATION_PLAN.md`](docs/TS_INLINE_HOST_INTEGRATION_PLAN.md),
[`docs/inline-natlang-lambdas.md`](docs/inline-natlang-lambdas.md), and
[`docs/ITERATE_ON_PLAN.md`](docs/ITERATE_ON_PLAN.md). `PLAN.md` and `plans/`
hold research history; the specification and the source are the active
contract.
