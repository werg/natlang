# natlang

natlang lets you automate work you would hand to an AI agent with a program you own, running on a small local
model. TypeScript holds the types, the data and every exact step; each judgment is a typed natural-language
function. Finite judgments are scored decisions with probabilities, outside text is fenced by its type, results are
checked against their declared properties, repeated paths compile to plain code, and the program improves from its
own runs.

<!-- from: examples/triage/classify.nl -->
```yaml
---
description: Label one ticket using the rubric.
args:
  ticket: string
  rubric: string
returns: Label
readout: decision
---
Pick the label from rubric that fits ticket.
```

<!-- from: examples/triage/types.ts -->
```ts
export type Label = "billing" | "technical" | "spam";
/** What the report says about the urgent tickets. */
export type Summary = Is<string, "one paragraph that names the most severe problem first">;
```

<!-- from: examples/triage/triage.ts -->
```ts
import { decide, IterationLimitError, iterateOn, nl } from '@natlang/node';
import classify from './classify.nl';
import isUrgent from './is_urgent.nl';
...
export async function triage(tickets: string[], rubric: string): Promise<Report> {
  const decisions = await Promise.all(tickets.map(ticket => decide(classify, ticket, rubric)));
  // Only the tickets the local model is unsure of go to the stronger model.
  const labels = await Promise.all(decisions.map(({ value, confidence }, i) =>
    confidence >= 0.8 ? value : classify.on({ model: 'teacher' })(tickets[i]!, rubric)));
  const real = tickets.filter((_, i) => labels[i] !== 'spam');
  const flags = await Promise.all(real.map(ticket => isUrgent(ticket)));
  const urgent = real.filter((_, i) => flags[i]);
  const by_label: Record<string, number> = {};
  for (const label of labels) by_label[label] = (by_label[label] ?? 0) + 1;
  return { urgent: urgent.length, by_label, summary: urgent.length ? await summarize(urgent) : 'Nothing urgent today.' };
}
...
  const draft: Summary = await nl`Write one paragraph saying what is going wrong across tickets, most severe first.`(tickets);
```

`classify.nl` is a natural-language function with `readout: decision`: the model scores the three labels in one pass,
and `decide` returns their probabilities. The tickets it is sure of are settled locally; `classify.on({ model:
'teacher' })` asks a stronger model about the rest (`teacher` is the profile that your model profile names as
`escalate`; without one the call runs on the default model). Counting is code. The summary's type is a property the
runtime checks on every draft.

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

### Install

natlang needs Node.js 22.13 or newer. Install the `natlang` command:

```sh
curl -fsSL https://raw.githubusercontent.com/werg/natlang/main/scripts/install.sh | sh
```

The script installs the `@natlang/node` and `@natlang/cli` packages of the newest release into npm's global prefix
(`~/.local` when that is not writable; `NATLANG_PREFIX` chooses another) and prints the `export PATH=...` line if the
command is not on your `PATH`. `natlang doctor` checks the installation.

To work on natlang itself, or to run the applications in this repository, install from a clone instead:

```sh
git clone https://github.com/werg/natlang.git && cd natlang
scripts/setup_dev.sh --node-only
```

This builds the runtime and links `~/.local/bin/natlang` to the clone's `scripts/natlang` (`--command-dir DIR`
chooses another directory).

### Your first program

Work in an empty directory outside the clone:

```sh
natlang init my-program && cd my-program
natlang check                          # the program, without a model
natlang setup                          # the managed local model, once (a 1.8 GB download)
natlang run                            # plan the bundled messages.json
natlang test                           # the saved route.nl cases on the model
natlang dev                            # reruns after source or input edits; Ctrl+C stops
```

`natlang init` writes an inbox planner, the paradigm in one screen. `types.ts` holds the data: a `Message` whose body
is `Untrusted<string>` (fenced as data, never followed) and a `Reply` typed `Is<string, "...">` (a property the runtime
checks on every draft). `route.nl` is a typed function with `readout: decision`: the model scores `reply`, `fyi` and
`ignore` in one pass. `main.ts` maps it over the messages with `decide(route, message)`, keeps each confidence, and
drafts a reply with an inline `nl<Reply>` only where one is needed. `examples.json` holds the `route.nl` cases,
`messages.json` the sample input, `natlang.json` the run target, and `AGENTS.md` (with `CLAUDE.md`) guides coding
agents. `natlang init DIR --template classifier` writes the one-function program instead: `answer.nl`, a three-label
classifier.

`natlang setup` is optional: for a hosted model, configure a profile on the dashboard's Models page
(`natlang dashboard --open`) or pass `--provider ID --model ID`. The CLI can use its managed local model (by default
the stock `LiquidAI/LFM3-1.7B`), an OpenAI-compatible endpoint, or a Pi provider with an API key or supported
subscription login. While calls run, the terminal shows which function is working, its turn and token counts; a call
that fails is always reported. See [model configuration](NATIVE_PACKAGES.md#authority-and-the-model-runtime) for
profiles and overrides.

### A second program

[`examples/review`](examples/review/) in the clone is the next step up: a named function whose callable folder holds
a natural-language helper (`.nl`) with its own typed decision (`readout: decision`) and an exact TypeScript helper
(`.ts`), with its manifest, saved examples that test the decisions, and a README.

```sh
natlang run examples/review            # from the clone
```

### When a result is wrong

Every model call gets an ID; `natlang call` prints it on stderr (`natlang: call CALL_ID (natlang traces show CALL_ID)`),
and `natlang test --json` reports one per example.

```sh
natlang traces show CALL_ID                              # inputs, each turn, code, output
natlang traces fix CALL_ID --expected '"technical"'     # saves the right answer into examples.json
natlang test                                             # now checks that case too
```

Then edit the instructions until `natlang test` passes. A unique prefix of a call ID is enough.

### With a coding agent

Run `natlang skills install` once to add the natlang skills to Claude Code or Codex; a new program's `AGENTS.md`
points the agent to them. Every command an agent needs is non-interactive: `natlang check --json`,
`natlang call ... --json` (the bare result on stdout), `natlang test --json` and `natlang setup --yes`. `check`, `call`,
`test` and `run` exit with:

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | The program is wrong: diagnostics, failed examples, or a call that failed for model-work reasons |
| 2 | Usage error: bad flags, arguments or paths |
| 3 | The model or runtime is unavailable: not installed, endpoint unreachable, missing credentials |

The dashboard (`natlang dashboard --open`) creates programs, edits function source, saves tests, and turns a
Playground experiment into a program. The [user-testing guide](docs/USER_TESTING.md) is a short walkthrough.

## Read next

### Using natlang

Read these programs first; each is the paradigm in working code:

| Program | What it shows |
|---|---|
| [`examples/triage`](examples/triage/) | Scored labels with escalation of the unsure ones, exact counts, a summary refined to its type |
| [`examples/review`](examples/review/) | A named function whose callable folder holds a decision leaf and an exact aggregate |
| [`examples/refunds`](examples/refunds/) | One verdict function bound per shop with `.in(context)`, a `pluggable` rule, a reply revised until a natural-language check holds, a folder transaction |
| [`applications/notebook`](applications/notebook/) | A dependency graph run one ready cell per step under a measure, with checked citations |
| [`applications/helpdesk`](applications/helpdesk/) | Per-ticket event loops whose policies are natural-language functions behind settings |
| [`applications/nldb`](applications/nldb/) | A relational database whose engine is natural-language stages over a folder format |

The authoring skill's [composition reference](skills/natlang-authoring/references/composition.md) walks through them.

| Guide | Contents |
|---|---|
| [Running programs](docs/RUNNING_PROGRAMS.md) | The default model, profiles, progress, compiled programs with warm starts, the dashboard |
| [Language specification](spec/SPEC.md) | Functions, callable folders, captures, iteration, services, the model surface |
| [Skills](skills/README.md) | Authoring and integration guidance for coding agents |
| [Native packages](NATIVE_PACKAGES.md) | `@natlang/node`, `@natlang/browser`, the CLI, and distribution packages |
| [TypeScript runtime](ts-host/README.md) | Building and using the runtime and compiler |
| [Development setup](DEV_SETUP.md) | Working on natlang itself: the browser build and the tests |
| [Tuning](docs/TUNING.md) | Specializing a student model to one program |
| [Adaptation](docs/ADAPTATION.md) | Evaluation suites, native GEPA/reflection, resumable runs, artifact deployment and rollback |
| [Applications](applications/) | Wiki, notebook, evidence, logs, terminal, publisher, games, and more (run them from a clone) |

### Research and training

| Guide | Contents |
|---|---|
| [Our trained models](docs/NATIVE_MODELS.md) | Pin a trained checkpoint, native server or GGUF, prefix reuse, escalation, feedback to tuning |
| [Research notes](docs/research/) | Design plans, experiment reports and the historical training pipeline |
| [Training](docs/research/TRAINING.md), [teacher setup](docs/research/TEACHER_SETUP.md), [program IR](docs/research/PROGRAM_IR_PIPELINE.md) | Teacher data, student training, and evaluation |
| [`plans/`](plans/) | Current research plans; `plans/neuralese/` holds the model training work |

The design notes behind the TypeScript-native runtime are
[the integration plan](docs/research/TS_INLINE_HOST_INTEGRATION_PLAN.md),
[inline lambdas](docs/research/inline-natlang-lambdas.md) and
[`iterateOn`](docs/research/ITERATE_ON_PLAN.md). The specification and the source are the active contract.
