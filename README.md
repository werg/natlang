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

[natlang, explained](https://werg.github.io/natlang/) is an illustrated, interactive
introduction in nine chapters; its source is in [`website/`](website/).

## Get started

```sh
scripts/setup_dev.sh --node-only        # install the CLI; no model download required
natlang run examples/hello              # one input, one natural-language function, one label
natlang dashboard --root examples --open # browse examples, configure models, run and inspect results
```

From a fresh terminal, clone this repository and enter its directory before running setup.
The first local run prepares the runtime and model once (the default model is a
1.8 GB download). Run `natlang setup` to prepare it in advance. For a hosted
model, configure a profile on the dashboard's Models page before your first run.
The next example is `natlang run examples/triage`, which processes bundled support tickets.

Create your own program:

```sh
natlang init my-program
cd my-program
natlang run                            # uses sample.json
natlang dev                            # reruns after source or input edits; Ctrl+C stops
natlang test                           # compares saved examples with real model results
```

Edit `answer.nl` to change instructions, `sample.json` to change input, and
`examples.json` to save expected results. `natlang check` checks source without
contacting a model. The dashboard can create programs, edit function source,
save tests, and turn a Playground experiment into a program. See the
[user-testing guide](docs/USER_TESTING.md) for a short walkthrough.

`natlang setup` prepares the selected model backend. The CLI can use its managed
local model (by default the stock `LiquidAI/LFM3-1.7B`), an OpenAI-compatible endpoint,
or a Pi provider with an API key or supported subscription login; the dashboard's
Models page configures all three. While natural-language calls run, `natlang run`
shows which function is working, its turn and token counts; a call that fails is
always reported. See
[model configuration](NATIVE_PACKAGES.md#authority-and-the-model-runtime)
for profiles, overrides, and provider options; see
[development setup](DEV_SETUP.md) for the browser build and tests.

## Read next

| Guide | Contents |
|---|---|
| [Running programs](docs/RUNNING_PROGRAMS.md) | The default model, profiles, progress, compiled programs with warm starts, the dashboard |
| [Language specification](spec/SPEC.md) | Functions, callable folders, captures, iteration, services, the model surface |
| [Skills](skills/README.md) | Authoring and integration guidance for coding agents |
| [Native packages](NATIVE_PACKAGES.md) | `@natlang/node`, `@natlang/browser`, the CLI, and distribution packages |
| [TypeScript runtime](ts-host/README.md) | Building and using the runtime and compiler |
| [Our trained models](docs/NATIVE_MODELS.md) | Pin a trained checkpoint, native server or GGUF, prefix reuse, escalation, feedback to tuning |
| [Adaptation](docs/ADAPTATION.md) | Evaluation suites, native GEPA/reflection, resumable runs, artifact deployment and rollback |
| [Applications](applications/) | Wiki, notebook, evidence, logs, terminal, publisher, games, and more |
| [Training](TRAINING.md), [teacher setup](TEACHER_SETUP.md), [program IR](PROGRAM_IR_PIPELINE.md) | Teacher data, student training, and evaluation |

The design notes behind the TypeScript-native runtime are in
[`docs/TS_INLINE_HOST_INTEGRATION_PLAN.md`](docs/TS_INLINE_HOST_INTEGRATION_PLAN.md),
[`docs/inline-natlang-lambdas.md`](docs/inline-natlang-lambdas.md), and
[`docs/ITERATE_ON_PLAN.md`](docs/ITERATE_ON_PLAN.md). `PLAN.md` and `plans/`
hold research history; the specification and the source are the active
contract.
