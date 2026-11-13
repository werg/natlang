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

[Try natlang in your browser](https://werg.github.io/natlang/play/): the example
gallery, an editor and a small model that runs in the page, with nothing to install.

To install the `natlang` command (it needs Node.js 22.13 or newer):

```sh
curl -fsSL https://raw.githubusercontent.com/werg/natlang/main/scripts/install.sh | sh
natlang init my-program && cd my-program
natlang setup                          # the managed local model, once (a 1.8 GB download)
natlang run
```

The script installs the `@natlang/node` and `@natlang/cli` packages of the newest
release with npm, into npm's global prefix or `~/.local`. `natlang setup` is
optional: for a hosted model, configure a profile on the dashboard's Models page
(`natlang dashboard --open`) or pass `--provider`. To work on natlang itself, or to
run the bundled examples (`natlang run examples/hello`, `natlang run examples/triage`),
clone this repository and run `scripts/setup_dev.sh --node-only`
([development setup](DEV_SETUP.md)).

Inside a program:

```sh
natlang run                            # answer.nl on the sample input
natlang call answer.nl "Your text"     # one call with your own input
natlang dev                            # reruns after source or input edits; Ctrl+C stops
natlang test                           # compares saved examples with real model results
```

A new program is one function (`answer.nl`), its manifest (`natlang.json`, whose
`main` target runs `answer.nl` with the sample as the default input), saved tests
(`examples.json`) and an `AGENTS.md` for coding agents. Edit `answer.nl` to change
instructions and `examples.json` to save expected results. `natlang check` checks source without
contacting a model. With a coding agent, run `natlang skills install` once to add the
natlang skills to Claude Code or Codex; every command an agent needs is
non-interactive (`natlang check --json`, `natlang call ... --json`,
`natlang test --json`, `natlang setup --yes`). The dashboard can create programs, edit function source,
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
| [Training](docs/research/TRAINING.md), [teacher setup](docs/research/TEACHER_SETUP.md), [program IR](docs/research/PROGRAM_IR_PIPELINE.md) | Teacher data, student training, and evaluation |

The design notes behind the TypeScript-native runtime are in
[`docs/research/TS_INLINE_HOST_INTEGRATION_PLAN.md`](docs/research/TS_INLINE_HOST_INTEGRATION_PLAN.md),
[`docs/research/inline-natlang-lambdas.md`](docs/research/inline-natlang-lambdas.md), and
[`docs/research/ITERATE_ON_PLAN.md`](docs/research/ITERATE_ON_PLAN.md). `docs/research/PLAN.md` and `plans/`
hold research history; the specification and the source are the active
contract.
