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
  const technical = tickets.filter((_, i) => labels[i] === 'technical');
  const summary: string = await nl`Summarize what is going wrong across technical, most severe first.`(technical);
  return { technical: technical.length, summary };
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
natlang setup                          # the managed local model, once (a 1.8 GB download)
natlang run                            # answer.nl on the sample input
natlang call answer.nl "Your text"     # one call with your own input
natlang dev                            # reruns after source or input edits; Ctrl+C stops
natlang test                           # runs the saved examples on the model
natlang check                          # checks the source without a model
```

`natlang init` writes `answer.nl` (the function), `natlang.json` (the manifest; its `main` target runs `answer.nl`
with the sample as the default input), `examples.json` (saved tests), `README.md`, `AGENTS.md` and `CLAUDE.md` for
coding agents, `.gitignore` and `package.json`. `answer.nl` is a classifier with `readout: decision`: the model
scores the three labels instead of writing an answer, which takes seconds rather than minutes on a CPU. Edit its instructions to change
behaviour and `examples.json` to save expected results.

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
