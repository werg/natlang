# pi on natlang

A port of [pi](https://github.com/earendil-works/pi)'s durable coding agent harness (pi-durable) to natlang, unit by
unit (PORT.md has every decision; `port/` is the inventory). pi-durable stays the host for what is mechanism: the
Session line and SQLite storage, the scheduler's invocations and timers, the registry and the Harness API. Everything
that decides what happens in a run is a natural-language function, run by natlang's executor (a small, fast model):

- each phase of the three durable task kinds: `generation.nl` (prepare, request, classify, answer, the tool round,
  abort), `tool.nl` (lookup, repair, validation, hooks, intent, run, settlement, recovery) and `compaction.nl`
  (the cut, the summary, its placement);
- context derivation, the system prompt plan, the compaction cut and the token estimate, steering and follow-up
  selection at turn boundaries;
- admission (`admit.nl`) and the scheduler's policy (`scheduler/`: which task runs, step precedence, abort and its
  cascades, finalization, idle);
- pi's coding tools (`extensions/coding-tools`: read, write, edit, bash) and the subagent.

The agent is a provider model the harness calls through pi-ai (`ai.turn`). The functions are the harness around it;
none of them is the agent.

```
index.ts                 openPi(): pi-durable's Harness with the natural-language task kinds
main.ts                  print-mode CLI and eval over tasks/
types.ts                 every record the functions read and write, with its rules in doc comments
ops.ts                   durable.commit: the write operations, applied atomically with guards
host/                    services: durable, ai, tools, env, resources; the task kinds; policies
generation.nl generation/  prepare, request, classify, answer, startToolRound, finishToolRound, abort
tool.nl tool/            beginCall, run (+ exact result formats), fromSlot
compaction.nl compaction/  select, summarize (+ pi's verbatim prompts and transcript format)
harness/                 shared through uses:: context, planSystem, estimate (pluggable), deriveContext, cut, boundary
admit.nl scheduler/      admission; pass, step, reconcile, abortTask, abortConversation, cleanup
extensions/              coding-tools, pi-prompt (pi's seven sections, verbatim), subagent
vendor/durable           pi-durable at f10993b (PATCHES.md lists the changes)
test/                    scripted wiring tests; conformance/ runs pi-durable's own suites against the port
tasks/                   coding tasks for eval
```

## How a function commits

Natural language never runs inside a commit. A function reads committed state through `durable`, decides, and hands
the host one list of write operations with guards on what it read: `durable.commit(ops, expect)`. The host applies the
list atomically on the Session line with pi-durable's own helpers, so their invariants hold (every assistant entry's
usage is counted, a stored tool result is exactly what the model sees, a run's inputs settle with the run). A failed
guard writes nothing and rejects with "state changed"; the function reads again and decides again. The rules that
judge a commit's own staged writes (hold or terminate, wait validation, the runtime commit gates) stay in the host.

## Variants

Context building, the system-entry plan with the context estimate, the scheduler's policy and admission run on every
turn and every phase. Each has pi-durable's crisp code and the natural-language functions behind one interface:

- default: the four are crisp; everything else is natural language;
- `--context natural-language`, `--scheduler natural-language`, `--admission natural-language`: one at a time;
- `--planning natural-language`: the system-entry plan and the context estimate (prepare runs both every turn);
- `--pure`: all four in natural language, so every function in this app runs.

## Running

```sh
natlang run --profile pi-executor applications/pi -- \
  --agent-endpoint http://127.0.0.1:8083 --agent-model nvidia/Qwen3.6-35B-A3B-NVFP4 "Fix the failing test"
natlang run --profile pi-executor applications/pi -- eval js-off-by-one --agent-endpoint … --agent-model …
```

The launcher's profile is the executor. The agent is any OpenAI-compatible server (`--agent-endpoint`,
`--agent-model`, `--agent-key-env`), registered with pi-ai as provider `agent`; without them the agent uses the
launcher's endpoint (the `--profile` it runs with), else the default profile's. `--session FILE` keeps the SQLite session; `--thinking LEVEL` sets the agent's thinking
level; `--quiet` hides the phase log.

## Verification

- `npm test` in this directory: scripted wiring tests (25). The crisp helpers are compared with pi's own code on
  thousands of random inputs (result bounding, truncation, transcript serialization, edit matching, image sniffing);
  the coding tools run through the runtime and are compared with pi's `CodingTools` (results, files, diagnostics);
  the policy path drives a real Harness through the scheduler and admission functions.
- `npm run test:vendor`: pi-durable's own suite with the crisp defaults (967 tests; three files need other packages
  of pi's monorepo and do not load). `npm run test:policy`: the same suite through the guarded read-decide-commit
  policy path.
- `npm run test:conformance`: pi-durable's harness suites (generation, generation-recovery, compaction, context,
  prompt, inbox, submissions, tools, tools-recovery, structured, tasks) with the natural-language task kinds
  substituted, against a real executor (`PI_EXECUTOR_ENDPOINT`, `PI_EXECUTOR_MODEL`; `PI_CONTEXT`, `PI_SCHEDULER`,
  `PI_ADMISSION` select natural-language implementations). The suites check exact entries and documents, so they test
  whether the functions are exact. Results: see Status.
- `eval` runs the coding tasks in `tasks/` on fresh git copies, judged by their check commands.

## Status

Measured on DGX (2026-10-08) with Qwen3.6-35B-A3B-NVFP4 on vLLM as both executor and agent, sharing the GPU with
a training job (about 170 generated tokens/s across all requests, about 28 per stream).

- Wiring tests: 25/25. pi-durable's own suite (crisp defaults): 967 tests.
- Conformance (pi-durable's harness suites against the natural-language task kinds, 322 tests): run 3 in
  progress; 113 of the first 121 pass. Every failure seen so far either passes alone on the current code or led
  to a fix: a string ID in a checkpoint, a repeated provider request, an empty system entry, a non-JSON provider
  message cleaned instead of faulting, an invented settlement status, the abort phase skipping its commit, a
  generation turn not streaming, the reported error's shape. Most fixes are commit-time checks in `ops.ts` whose
  errors say what to do, or general runtime changes (see below). Remaining variance comes from the executor under
  load: a phase occasionally improvises (skips a step, retypes a value) and the second attempt, which is told why
  the first failed (`facts.previousAttempt`), usually recovers.
- Results of an aborted parallel round are written in call order: an aborted tool waits (bounded) for the slots
  before it, as pi's synchronous tools finish in call order.
- Live: `eval js-off-by-one` ran 7 generations and several tool rounds end to end (bash, read, edit), fixed the
  visible tests and missed the hidden `hasNext` case before its 90-minute limit. The subagent tool ran a child
  conversation end to end through `delegation` (the child ran bash, the parent answered with its findings).
- Speed: a phase is a multi-turn executor call, 1 to 5 minutes under this load. Planning (planSystem and the
  token estimate) has crisp twins behind the hot-path setting (`--planning`, default crisp), which cut prepare
  from 4–7 minutes to about 1.5.

General runtime changes that came out of the port: service arguments arrive as host-realm data; `ONCE_EFFECTS`
(the provider request and poll run once per phase); computed values keep fields their declared type does not
list, in returns and locals (the cause of executors retyping provider messages); union type errors name the
field; the default context budget follows the server's window; `natlangVitePlugin()`.
